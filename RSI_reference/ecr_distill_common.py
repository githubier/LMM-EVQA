"""Shared infrastructure for ECR rubric distillation."""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from pydantic import BaseModel, conlist


def constrained_list(item_type, minimum, maximum=None):
    """Create a conlist whose JSON Schema has minItems/maxItems on Pydantic v1/v2."""
    try:
        return conlist(item_type, min_length=minimum, max_length=maximum)
    except TypeError:  # Pydantic v1
        return conlist(item_type, min_items=minimum, max_items=maximum)


try:
    from pydantic import field_validator as compatible_validator
except ImportError:  # Pydantic v1
    from pydantic import validator as compatible_validator


def pydantic_schema(model_cls):
    """Pydantic v1/v2 compatibility."""
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


def pydantic_validate(model_cls, value):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(value)
    return model_cls.parse_obj(value)


def pydantic_dump(instance):
    if hasattr(instance, "model_dump"):
        return instance.model_dump()
    return instance.dict()

sys.path.append("/share/bizpulse/src/bizpulse/DBDE")
from llm_serving import llm_serving_langbridge_with_json_schema_outside

def load_optional_dotenv():
    """Keep the original ../.env behavior without making dotenv mandatory."""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path("..") / ".env", verbose=True)
    except ImportError:
        pass


ROLE = (
    "你是一个短视频用户行为分析专家。你的任务不是给单条视频打分，"
    "而是从一批带真实 ECR 标签的样本里，归纳出一套通用、可复用的评分规则，"
    "让另一个没有见过这批数据的模型也能照着规则去预测新视频的 ECR。"
    "ECR = 观看时长超过5秒的用户占比，范围 [0, 1]。"
)


def write_json(path, data):
    """Atomic JSON write."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_samples(path, configured_cols):
    """Load Id/Title/Description/Download_link/ECR with case-insensitive headers."""
    encoding = "utf-8"
    for candidate in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
        try:
            with open(path, encoding=candidate, newline="") as probe:
                probe.read(4096)
            encoding = candidate
            print("  csv encoding detected: {}".format(encoding), flush=True)
            break
        except UnicodeDecodeError:
            continue

    with open(path, encoding=encoding, errors="replace", newline="") as stream:
        reader = csv.reader(stream)
        try:
            first = next(reader)
        except StopIteration:
            raise ValueError("empty CSV: {}".format(path))
        first_lower = [x.strip().lower() for x in first]
        has_header = "ecr" in first_lower or "title" in first_lower
        headers = first if has_header else [x.strip() for x in configured_cols.split(",")]
        rows = list(reader)
        if not has_header:
            rows.insert(0, first)
    index = {name.strip().lower(): i for i, name in enumerate(headers)}

    def get(row, name):
        pos = index.get(name.lower())
        return row[pos].strip() if pos is not None and pos < len(row) else ""

    result = []
    seen = set()
    for row_no, row in enumerate(rows, 2 if has_header else 1):
        try:
            ecr = float(get(row, "ECR"))
        except (TypeError, ValueError):
            continue
        raw_id = get(row, "Id")
        uid = raw_id or f"row_{row_no}"
        if uid in seen:
            uid = f"{uid}__row_{row_no}"
        seen.add(uid)
        result.append({
            "id": uid, "source_id": raw_id,
            "title": get(row, "Title"),
            "description": get(row, "Description"),
            "url": get(row, "Download_link"), "ecr": ecr,
        })
    if not result:
        raise ValueError("no rows with valid ECR labels")
    return result


def probe_url(url, timeout):
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code == 405 or r.status_code >= 400:
            r = requests.get(url, timeout=timeout, stream=True,
                             headers={"Range": "bytes=0-1023"})
        if r.status_code >= 400:
            return False
        content_type = (r.headers.get("Content-Type") or "").lower()
        return not content_type or any(x in content_type for x in
            ("video/", "application/", "binary/", "mp4"))
    except Exception:
        return False


def filter_alive(samples, want_k, workers=32, timeout=5,
                 initial_batch=None, max_probe=None, exclude_ids=None):
    """Original adaptive probing logic: expand by observed death rate."""
    exclude_ids = exclude_ids or set()
    candidates = [s for s in samples
                  if s.get("url") and s["id"] not in exclude_ids]
    max_probe = len(candidates) if max_probe is None else min(max_probe, len(candidates))
    initial_batch = initial_batch or want_k * 3
    alive = []
    probed_total = 0
    batch = initial_batch

    while probed_total < max_probe and len(alive) < want_k:
        chunk = candidates[probed_total:probed_total + batch]
        if not chunk:
            break
        with ThreadPoolExecutor(max_workers=workers) as pool:
            flags = list(pool.map(lambda x: probe_url(x["url"], timeout), chunk))
        alive.extend(x for x, ok in zip(chunk, flags) if ok)
        probed_total += len(chunk)
        remaining = want_k - len(alive)
        print("    probed {}, alive so far {}/{}".format(
            probed_total, len(alive), want_k), flush=True)
        if remaining <= 0:
            break
        alive_rate = max(float(len(alive)) / probed_total, 0.05)
        batch = min(
            int(remaining / alive_rate * 1.5) + want_k,
            max_probe - probed_total,
        )
        batch = max(batch, want_k)

    # pool.map preserves candidate order, therefore ECR extreme order is preserved.
    return alive[:want_k], probed_total, len(alive)


def select_group(high_to_low, low_to_high, count_each, used_ids, args):
    if args.mode == "text":
        high = [x for x in high_to_low if x["id"] not in used_ids][:count_each]
        blocked = used_ids | {x["id"] for x in high}
        low = [x for x in low_to_high if x["id"] not in blocked][:count_each]
    else:
        high, high_probed, high_alive = filter_alive(
            high_to_low, count_each,
            workers=args.probe_workers, timeout=args.probe_timeout,
            initial_batch=count_each * args.over_sample,
            max_probe=args.max_probe, exclude_ids=used_ids)
        blocked = used_ids | {x["id"] for x in high}
        low, low_probed, low_alive = filter_alive(
            low_to_high, count_each,
            workers=args.probe_workers, timeout=args.probe_timeout,
            initial_batch=count_each * args.over_sample,
            max_probe=args.max_probe, exclude_ids=blocked)
        print("    HIGH: {}/{} alive, kept {}".format(
            high_alive, high_probed, len(high)), flush=True)
        print("    LOW : {}/{} alive, kept {}".format(
            low_alive, low_probed, len(low)), flush=True)
    return high, low


def sample_text(highs, lows, title_max, desc_max):
    lines = []
    for band, items in (("HIGH", highs), ("LOW", lows)):
        for x in items:
            lines.append(
                f"{band} id={x['id']} ECR={x['ecr']:.4f} | "
                f"标题:{x['title'][:title_max]} | 描述:{x['description'][:desc_max]}"
            )
    return "\n".join(lines)


def _strip_array_bounds(value):
    """Copy a JSON schema while removing minItems/maxItems for legacy serving."""
    if isinstance(value, dict):
        return {
            key: _strip_array_bounds(item)
            for key, item in value.items()
            if key not in ("minItems", "maxItems")
        }
    if isinstance(value, list):
        return [_strip_array_bounds(item) for item in value]
    return value


def repair_structured_output(model, original_task, invalid_output, schema_cls,
                             server_schema, schema_name, retries):
    """Repair a semantically incomplete JSON result using text-only calls."""
    counts = []
    for key in ("dimensions", "signals", "interactions",
                "recommended_call_groups"):
        value = invalid_output.get(key) if isinstance(invalid_output, dict) else None
        if isinstance(value, list):
            counts.append("{}={}".format(key, len(value)))
    repair_prompt = (
        "下面是一次多模态分析产生的 JSON 草稿，但未通过数量/结构校验（{}）。\n\n"
        "原任务要求：\n{}\n\nJSON 草稿：\n{}\n\n"
        "请只修复这个草稿，不要重新分析视频，不要解释。保留已有有效信息，"
        "将过度合并的维度按不同机制、时间窗口或模态拆分，并补齐独立的判分"
        "标准。输出完整 JSON，严格满足原任务的数量范围和全部必填字段。"
    ).format(
        ", ".join(counts) or "schema validation failed",
        original_task,
        json.dumps(invalid_output, ensure_ascii=False, indent=2),
    )
    messages = [
        {"role": "system", "content": ROLE + "\n\n你负责修复已有结构化分析结果。"},
        {"role": "user", "content": repair_prompt},
    ]
    last_error = None
    for attempt in range(1, retries + 1):
        answer = None
        try:
            _, answer, _ = llm_serving_langbridge_with_json_schema_outside(
                {"messages": messages}, model=model, server_url=None, headers=None,
                json_schema_name=schema_name or schema_cls.__name__,
                json_schema=server_schema,
            )
            if answer is None or (isinstance(answer, str) and not answer.strip()):
                raise ValueError("LLM serving returned an empty repair answer")
            parsed = json.loads(answer) if isinstance(answer, str) else answer
            return pydantic_dump(pydantic_validate(schema_cls, parsed))
        except Exception as exc:
            last_error = exc
            print("  repair attempt {}/{} failed: {}: {}".format(
                attempt, retries, type(exc).__name__, exc), flush=True)
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError("text-only repair failed after {} attempts: {}".format(
        retries, last_error))


def call_llm(model, system_text, user_text, schema_cls,
             videos=None, labels=None, retries=3, schema_name=None,
             legacy_server_schema=False, repair_retries=3):
    messages = [{"role": "system", "content": ROLE + "\n\n" + system_text}]
    if videos:
        content = [{"type": "text", "text": user_text}]
        for i, url in enumerate(videos):
            label = labels[i] if labels else f"sample {i + 1}"
            content.extend([
                {"type": "text", "text": f"[{label}]"},
                {"type": "video_url", "video_url": {"url": url}},
            ])
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_text})

    server_schema = pydantic_schema(schema_cls)
    if legacy_server_schema:
        server_schema = _strip_array_bounds(server_schema)
    last_error = None
    for attempt in range(1, retries + 1):
        answer = None
        try:
            _, answer, _ = llm_serving_langbridge_with_json_schema_outside(
                {"messages": messages}, model=model,
                server_url=None, headers=None,
                json_schema_name=schema_name or schema_cls.__name__,
                json_schema=server_schema,
            )
            if answer is None or (isinstance(answer, str) and not answer.strip()):
                raise ValueError("LLM serving returned an empty answer")
            parsed = json.loads(answer) if isinstance(answer, str) else answer
            try:
                return pydantic_dump(pydantic_validate(schema_cls, parsed))
            except Exception as validation_error:
                print("  multimodal output needs text-only repair: {}".format(
                    validation_error), flush=True)
                return repair_structured_output(
                    model, user_text, parsed, schema_cls, server_schema,
                    schema_name, repair_retries)
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            print("  LLM attempt {}/{} failed: {}: {}".format(
                attempt, retries, type(exc).__name__, exc), flush=True)
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError("LLM failed after {} attempts: {}".format(
        retries, last_error))


def video_payload(highs, lows):
    items = highs + lows
    videos = [x["url"] for x in items]
    labels = [
        f"{'HIGH' if i < len(highs) else 'LOW'} id={x['id']} ECR={x['ecr']:.4f}"
        for i, x in enumerate(items)
    ]
    return videos, labels


def ensure_manifest(out_dir, args, pipeline_type):
    """Refuse unsafe resume when data/model/prompt-affecting parameters changed."""
    identity = {
        "pipeline_type": pipeline_type,
        "input_file": str(Path(args.input_file).resolve()),
        "input_size": Path(args.input_file).stat().st_size,
        "model": args.model,
        "mode": args.mode,
        "videos_per_group": args.videos_per_group,
        "col_names": args.col_names,
        "title_max": args.title_max,
        "desc_max": args.desc_max,
        "reduce_batch_size": args.reduce_batch_size,
        "schema_version": "compact-v2-parity-audited",
    }
    identity["fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(out_dir) / "run_manifest.json"
    if args.resume:
        if not path.exists():
            raise ValueError(f"--resume requires existing manifest: {path}")
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != identity["fingerprint"]:
            completed = (list(Path(out_dir).glob("group*_rubric.json"))
                         + list(Path(out_dir).glob("atomic_group*.json")))
            # reduce_batch_size only changes the final aggregation strategy;
            # it is safe to change when reusing already completed Map groups.
            stable_keys = (
                "pipeline_type", "input_file", "input_size", "model", "mode",
                "videos_per_group", "col_names", "title_max", "desc_max",
                "schema_version",
            )
            stable_match = all(old.get(key) == identity.get(key)
                               for key in stable_keys)
            if completed and not stable_match:
                raise ValueError(
                    "resume manifest mismatch with completed groups; use a new out_dir. "
                    "old={} new={}".format(old, identity)
                )
            if completed:
                print("[resume] allowing reduce-only setting change: "
                      "reduce_batch_size {} -> {}".format(
                          old.get("reduce_batch_size"),
                          identity.get("reduce_batch_size")), flush=True)
            else:
                print("[resume] no completed groups; upgrading stale manifest", flush=True)
    else:
        # nohup/shell redirection may create run.log before Python starts.
        meaningful = [
            item for item in Path(out_dir).iterdir()
            if item.name != path.name and item.suffix.lower() not in (".log", ".out")
        ]
        if path.exists() or meaningful:
            raise FileExistsError(
                f"output directory contains experiment artifacts: {out_dir}; "
                "use a new directory or --resume"
            )
    write_json(path, identity)


def restore_state(out_dir, artifact_pattern, artifact_regex,
                  provenance_name, content_key, resume):
    """Restore only artifacts that have matching provenance entries."""
    if not resume:
        return [], [], set(), 1
    artifacts = {}
    for path in sorted(Path(out_dir).glob(artifact_pattern)):
        match = re.search(artifact_regex, path.name)
        if not match:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get(content_key):
                artifacts[int(match.group(1))] = data
        except Exception as exc:
            print(f"  skip invalid {path.name}: {exc}")

    prov_path = Path(out_dir) / provenance_name
    provenance = []
    if prov_path.exists():
        try:
            loaded = json.loads(prov_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                # Provenance is also the durable consumed-sample ledger. Keep
                # failed/pending attempts even when no rubric artifact exists.
                provenance = [x for x in loaded if isinstance(x, dict)
                              and isinstance(x.get("group"), int)]
        except Exception as exc:
            print(f"  ignore invalid provenance: {exc}")
    attempted_numbers = {x["group"] for x in provenance}
    artifacts = {n: data for n, data in artifacts.items()
                 if n in attempted_numbers}
    used = set()
    for entry in provenance:
        used.update(entry.get("high_ids", [])); used.update(entry.get("low_ids", []))
    ordered = [artifacts[n] for n in sorted(artifacts)]
    next_group = max(attempted_numbers | set(artifacts), default=0) + 1
    failed = sum(1 for x in provenance if x.get("status") == "failed")
    pending = sum(1 for x in provenance if x.get("status") == "pending")
    print("[resume] successful_groups={}, consumed_attempts={}, failed={}, "
          "pending={}, used_ids={}, next={}".format(
              len(ordered), len(provenance), failed, pending,
              len(used), next_group), flush=True)
    return ordered, provenance, used, next_group


def add_common_args(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--col_names", default="Id,Title,Description,Download_link,ECR")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=["text", "video_only", "video_text"], default="video_text")
    parser.add_argument("--n_groups", type=int, default=20)
    parser.add_argument("--videos_per_group", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe_workers", type=int, default=32)
    parser.add_argument("--probe_timeout", type=float, default=5.0)
    parser.add_argument("--over_sample", type=int, default=3,
                        help="首批探活过采样倍数；不足时按死亡率自适应补采")
    parser.add_argument("--max_probe", type=int, default=5000,
                        help="每个 HIGH/LOW 桶每组最多探查的候选数")
    parser.add_argument("--title_max", type=int, default=120)
    parser.add_argument("--desc_max", type=int, default=200)
    parser.add_argument("--max_retries", type=int, default=3)
    return parser
"""
Distill a reusable, model-agnostic rubric for predicting a short-video's
Engagement Continuation Rate (ECR = P(watch_time > 5s)) out of a fine-tuned
VideoLLaMA2 regressor.

Idea
----
The fine-tuned model outputs a scalar ECR. We can't just ask it "what do you
look at?" and expect a good answer — its head is a regressor, not a chat head.
So we combine two signals:

  (A) Self-report via generation head
      Switch the LM head back on (skip the regression head) and prompt the
      base LM with a meta-question asking it to enumerate the dimensions it
      uses when scoring ECR. Output a strict JSON schema.

  (B) Data-driven contrastive mining
      Take the training/eval json (the same file used by the original script),
      bucket samples by ECR (top-K vs bottom-K), then ask the LM head to
      compare the two buckets and abstract common features. This grounds the
      rubric in the actual distribution the model was trained on.

The final artifact is a JSON rubric + a ready-to-paste Markdown prompt you can
drop into any other model to score new videos zero-shot.

Usage
-----
uv run distill_ecr_rubric.py \
    --model-path /path/to/finetuned_videollama2 \
    --json-file  /path/to/train_or_val.json \
    --modal-type av \
    --top-k 30 --bottom-k 30 \
    --out-dir ./rubric_out
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append("./")
from videollama2 import model_init
from videollama2.mm_utils import tokenizer_multimodal_token


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

META_PROMPT_SELF_REPORT = """You are the very model that has been fine-tuned to predict the Engagement
Continuation Rate (ECR) of short videos, where ECR = probability that a viewer
keeps watching past 5 seconds. Your training signal was a scalar regression on
(video, title, description) -> ECR.

Task: introspect on what you have learned and produce a GENERAL, REUSABLE
rubric that another model (which has NOT been trained on this task) could
follow to estimate ECR from a new video's title, description, and visual/audio
content.

Constraints:
- Do NOT reference any single training example.
- Enumerate concrete, observable dimensions (not vague adjectives).
- For each dimension: (a) name, (b) what to look for, (c) directional effect
  on ECR (positive / negative / U-shaped), (d) approximate weight
  (high / medium / low), (e) a 1-line detection heuristic another LLM/VLM can
  actually execute.
- Cover at least: opening-second hook, visual novelty, motion/pace,
  audio hook, subject clarity, cognitive payoff / curiosity gap, aesthetic
  quality, text-overlay strategy, title-content congruence, emotional
  valence/arousal, familiar-vs-novel framing, platform/format fit.
- End with a scoring recipe: how to combine the per-dimension scores into a
  single ECR estimate in [0, 100].

Output STRICT JSON with this schema and nothing else:

{
  "dimensions": [
    {
      "name": "...",
      "what_to_look_for": "...",
      "direction": "positive|negative|u_shaped",
      "weight": "high|medium|low",
      "detection_heuristic": "..."
    }
  ],
  "scoring_recipe": "...",
  "caveats": ["...", "..."]
}
"""


CONTRASTIVE_PROMPT_TEMPLATE = """You are the fine-tuned ECR predictor. Below are two groups of real videos
from your training distribution, described by their titles and descriptions,
already labeled with true ECR values.

GROUP HIGH (top {k_high} by ECR):
{high_block}

GROUP LOW (bottom {k_low} by ECR):
{low_block}

Compare the two groups and abstract the transferable patterns that separate
HIGH from LOW. Do not overfit to any single item. Produce STRICT JSON:

{{
  "high_ecr_patterns":  [{{"pattern": "...", "why_it_works": "..."}}, ...],
  "low_ecr_patterns":   [{{"pattern": "...", "why_it_hurts":  "..."}}, ...],
  "confounds_to_ignore":[ "..." ]
}}
"""


MERGE_PROMPT_TEMPLATE = """Merge the following two artifacts into ONE final, portable rubric that a
different (untrained) model can follow to score ECR from a new video.

--- SELF_REPORT ---
{self_report}

--- CONTRASTIVE ---
{contrastive}

Requirements for the merged rubric:
- 8-14 dimensions, deduplicated.
- Each dimension has: name, what_to_look_for, direction, weight_0_1
  (a float summing across all dimensions to 1.0), detection_heuristic.
- Add a "scoring_formula" field: an explicit weighted-sum formula on
  0-1 per-dimension scores, then a mapping to ECR in [0, 100].
- Add "few_shot_calibration": 3 hypothetical (title, description, expected
  ECR band) tuples that anchor the scale.

Output STRICT JSON only.
"""


# --------------------------------------------------------------------------- #
# Generation helpers
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 1200) -> str:
    """
    This fine-tuned checkpoint's forward is hard-wired to a regression path:
        hidden -> pooled -> lm_head_reg -> scalar
    so .generate() can never work as-is. We monkey-patch forward during
    generation to run the standard CausalLM path: hidden -> lm_head -> vocab
    logits, using the still-present original `lm_head`.
    """
    from transformers.modeling_outputs import CausalLMOutputWithPast
    lm = model

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # figure out device from the embed layer (accelerate may shard across GPUs)
    try:
        dev = lm.get_input_embeddings().weight.device
    except Exception:
        dev = next(lm.parameters()).device
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(dev)

    # This fork's generate signature is: generate(self, inputs, images_or_videos=..., modal_list=..., **kwargs)
    # Pass by keyword to avoid positional-slot mismatch.
    gen_kwargs = dict(
        inputs=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # try the common multimodal kwargs; ignore if the fork doesn't accept them
    import inspect
    sig = inspect.signature(lm.generate)
    if "images_or_videos" in sig.parameters:
        gen_kwargs["images_or_videos"] = None
    if "modal_list" in sig.parameters:
        gen_kwargs["modal_list"] = []
    if "images" in sig.parameters:
        gen_kwargs["images"] = None

    # --- monkey-patch forward to a plain CausalLM path during generation ---
    orig_forward = lm.forward
    backbone = lm.get_model() if hasattr(lm, "get_model") else lm.model

    def lm_forward(input_ids=None, attention_mask=None, position_ids=None,
                   past_key_values=None, inputs_embeds=None, labels=None,
                   use_cache=None, output_attentions=None,
                   output_hidden_states=None, return_dict=None, **kw):
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = outputs[0]
        logits = lm.lm_head(hidden).float()
        # accelerate may place lm_head on a different GPU than input_ids;
        # transformers' generate assumes logits live on the input device.
        if input_ids is not None:
            logits = logits.to(input_ids.device)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    lm.forward = lm_forward
    try:
        out = lm.generate(**gen_kwargs)
    finally:
        lm.forward = orig_forward
    # NOTE: this fork's `generate()` converts input_ids -> inputs_embeds before
    # calling super().generate(), so the returned tensor contains ONLY the newly
    # generated tokens — it is NOT prefixed with the prompt. Slicing off
    # `input_ids.shape[1]` here would silently chop the first N tokens of the
    # answer (which is why the previous rubric_out/*.json all started mid-JSON
    # and parsed to an empty `_raw` blob).
    gen = out[0]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def try_parse_json(s: str):
    """Best-effort JSON extraction from a possibly-noisy model output."""
    import re
    # grab the largest {...} block
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return {"_raw": s}
    blob = m.group(0)

    candidates = [blob]

    # trailing commas before a closing bracket
    candidates.append(re.sub(r",\s*([\]}])", r"\1", blob))

    # The 7B model frequently closes a string with a single quote instead of a
    # double quote (e.g. `"name": "Motion/pace',`) and drops the underscore in
    # keys (`"what_to look for"`). Repair both, then retry.
    repaired = re.sub(r'"\s*([^"\n]*?)\'(\s*[,:\]\}])', r'"\1"\2', blob)
    repaired = re.sub(
        r'"(what[_ ]to[_ ]look[_ ]for|detection[_ ]heuristic|weight[_ ]0[_ ]1)"',
        lambda mm: '"' + mm.group(1).replace(" ", "_") + '"',
        repaired,
    )
    repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
    candidates.append(repaired)

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return {"_raw": s}


# --------------------------------------------------------------------------- #
# Data-side mining
# --------------------------------------------------------------------------- #

def load_samples(json_file: str):
    with open(json_file, "r") as f:
        data = json.load(f)
    # each item has: id, ECR, video, audio, conversations
    # description lives inside the human turn value; parse it out.
    parsed = []
    for s in data:
        human = s["conversations"][0]["value"]
        title = None
        desc = None
        # "The title of the video is X, and the description of the video is Y"
        if "The title of the video is" in human:
            try:
                title = human.split("The title of the video is", 1)[1]
                title = title.split(", and the description of the video is", 1)[0].strip()
            except Exception:
                pass
        if "the description of the video is" in human:
            try:
                desc = human.split("the description of the video is", 1)[1].strip()
            except Exception:
                pass
        parsed.append({
            "id": s["id"],
            "ecr": float(s["ECR"]),
            "title": title,
            "description": desc,
        })
    return parsed


def format_bucket(items, header):
    lines = []
    for i, it in enumerate(items, 1):
        t = (it["title"] or "None").replace("\n", " ")[:120]
        d = (it["description"] or "None").replace("\n", " ")[:200]
        lines.append(f"  {i:>2}. ECR={it['ecr']:.2f} | title: {t} | desc: {d}")
    return f"{header}\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--json-file",  required=True,
                    help="Same json used for eval/train — used only for "
                         "contrastive mining (titles+descriptions+ECR).")
    ap.add_argument("--modal-type", default="av", choices=["a", "v", "av"])
    ap.add_argument("--top-k",    type=int, default=30)
    ap.add_argument("--bottom-k", type=int, default=30)
    ap.add_argument("--out-dir",  default="./rubric_out")
    ap.add_argument("--skip-contrastive", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/4] loading model…")
    model, processor, tokenizer = model_init(args.model_path)
    model.get_tokenizer(tokenizer)
    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    model.eval()

    # ---------- (A) self-report ----------
    print("[2/4] self-report generation…")
    self_report_raw = generate_text(model, tokenizer, META_PROMPT_SELF_REPORT,
                                    max_new_tokens=1500)
    self_report = try_parse_json(self_report_raw)
    (out_dir / "01_self_report.json").write_text(
        json.dumps(self_report, ensure_ascii=False, indent=2))
    (out_dir / "01_self_report.raw.txt").write_text(self_report_raw)

    # ---------- (B) contrastive mining ----------
    contrastive = None
    if not args.skip_contrastive:
        print("[3/4] contrastive mining from json_file…")
        samples = load_samples(args.json_file)
        samples.sort(key=lambda x: x["ecr"], reverse=True)
        highs = samples[:args.top_k]
        lows  = samples[-args.bottom_k:]

        prompt = CONTRASTIVE_PROMPT_TEMPLATE.format(
            k_high=len(highs), k_low=len(lows),
            high_block=format_bucket(highs, "HIGH"),
            low_block =format_bucket(lows,  "LOW"),
        )
        contrastive_raw = generate_text(model, tokenizer, prompt,
                                        max_new_tokens=1500)
        contrastive = try_parse_json(contrastive_raw)
        (out_dir / "02_contrastive.json").write_text(
            json.dumps(contrastive, ensure_ascii=False, indent=2))
        (out_dir / "02_contrastive.raw.txt").write_text(contrastive_raw)
    else:
        print("[3/4] skipping contrastive mining (per flag)")

    # ---------- (C) merge into final rubric ----------
    print("[4/4] merging into final rubric…")
    merge_prompt = MERGE_PROMPT_TEMPLATE.format(
        self_report=json.dumps(self_report, ensure_ascii=False, indent=2),
        contrastive=json.dumps(contrastive or {}, ensure_ascii=False, indent=2),
    )
    merged_raw = generate_text(model, tokenizer, merge_prompt,
                               max_new_tokens=1800)
    merged = try_parse_json(merged_raw)
    (out_dir / "03_final_rubric.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2))
    (out_dir / "03_final_rubric.raw.txt").write_text(merged_raw)

    # ---------- (D) ready-to-paste Markdown prompt ----------
    md = render_markdown_prompt(merged)
    (out_dir / "04_reusable_prompt.md").write_text(md)

    print(f"\n✅ Done. Artifacts in {out_dir.resolve()}")
    print("   - 01_self_report.json      (model introspection)")
    print("   - 02_contrastive.json      (data-grounded patterns)")
    print("   - 03_final_rubric.json     (merged, weighted rubric)")
    print("   - 04_reusable_prompt.md    (drop into any other model)")


def render_markdown_prompt(rubric: dict) -> str:
    """Turn the JSON rubric into a plug-and-play system prompt."""
    header = (
        "You are estimating the Engagement Continuation Rate (ECR) of a short "
        "video: the probability that a viewer keeps watching past 5 seconds. "
        "Follow the rubric below. Score each dimension in [0, 1], then apply "
        "the scoring formula, then output ECR in [0, 100]."
    )

    dims = rubric.get("dimensions")
    usable = (
        isinstance(dims, list)
        and dims
        # a "repaired" JSON can still be garbage (keys renamed / values lost);
        # require most entries to actually carry a name + description
        and sum(
            1
            for d in dims
            if isinstance(d, dict) and d.get("name") and d.get("what_to_look_for")
        )
        >= max(2, len(dims) // 2)
    )
    if not usable:
        # Ship the raw model text rather than an empty shell, otherwise
        # --rubric-file would inject a no-op prompt.
        raw = rubric.get("_raw") or json.dumps(rubric, ensure_ascii=False, indent=2)
        return "\n".join(
            ["# ECR Scoring Rubric (portable, unparsed)", "", header, "",
             "## Raw rubric text", "", raw]
        )

    lines = ["# ECR Scoring Rubric (portable)", "", header, "", "## Dimensions", ""]
    for d in dims:
        lines += [
            f"### {d.get('name','?')}  (weight={d.get('weight_0_1', d.get('weight','?'))}, "
            f"direction={d.get('direction','?')})",
            f"- **What to look for:** {d.get('what_to_look_for','')}",
            f"- **How to detect:** {d.get('detection_heuristic','')}",
            "",
        ]
    lines += [
        "## Scoring formula",
        "",
        "```",
        str(rubric.get("scoring_formula") or rubric.get("scoring_recipe") or ""),
        "```",
        "",
        "## Few-shot calibration",
        "",
        json.dumps(rubric.get("few_shot_calibration", []),
                   ensure_ascii=False, indent=2),
        "",
        "## Caveats",
        "",
    ]
    for c in rubric.get("caveats", []) or []:
        lines.append(f"- {c}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Mario OpenAPI 客户端 —— 调用你自己创建的 Agent。

用法：
    1. 在文件顶部填写 SYSTEM_PROMPT、USER_PROMPT、VIDEO_DIR、VIDEO_SELECTION_FILE
    2. 在视频清单文件中按顺序填写需要处理的视频文件名
    3. 执行：uv run mario_client.py <username> --env prt-idc

    <username>: 邮箱前缀 或 kwaiUserId

依赖：requests（uv run 会自动装）
"""
# /// script
# dependencies = ["requests"]
# ///

import json
import os
import sys
import argparse
from typing import List, Optional

import requests

# ── 调用配置：请在这里填写 ───────────────────────────────────────
API_KEY = "mak_xxxxxxxxxxxxxxxx"        # 平台申请的 Mario API Key
AGENT_ID = "mario-agent-xxxxxxxx"       # 你创建好的 Agent ID

# 留空待后续调优。调用时不再从命令行传入 prompt。
SYSTEM_PROMPT = ""
USER_PROMPT = ""

# 视频都放在这个目录下。清单文件决定本次读取哪些视频及输入顺序。
VIDEO_DIR = "/path/to/videos"
VIDEO_SELECTION_FILE = "/path/to/video_selection.txt"

# 清单文件支持两种格式：
# 1. 纯文本：每行一个相对于 VIDEO_DIR 的文件名，空行和 # 注释会忽略。
# 2. JSON：字符串数组，例如 ["video1.mp4", "video2.mp4"]；
#          或对象，例如 {"videos": ["video1.mp4", "video2.mp4"]}。

# ── 环境切换：默认 staging，可按需改 ──────────────────────────────
BASE_URLS = {
    "prod":    "http://mario.internal/mario/openapi",
    "prt-idc": "http://mario-open-prt.internal/mario/openapi",
    "prt":     "https://ee-mario.prt.kuaishou.com/mario/openapi",
    "staging": "https://kim-mario.staging.kuaishou.com/mario/openapi",
}


def load_selected_videos(video_dir: str, selection_file: str) -> List[str]:
    """从外部清单读取视频文件名，并安全解析为 VIDEO_DIR 下的绝对路径。"""
    base_dir = os.path.realpath(video_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"视频目录不存在或不是目录: {video_dir}")
    if not os.path.isfile(selection_file):
        raise ValueError(f"视频清单文件不存在: {selection_file}")

    with open(selection_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError("视频清单文件为空")

    if selection_file.lower().endswith(".json"):
        data = json.loads(raw)
        names = data.get("videos") if isinstance(data, dict) else data
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise ValueError("JSON 清单必须是字符串数组，或包含 videos 字符串数组的对象")
    else:
        names = [
            line.strip() for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not names:
        raise ValueError("视频清单中没有有效文件名")

    paths = []
    for name in names:
        if os.path.isabs(name):
            raise ValueError(f"清单中只允许相对路径，不允许绝对路径: {name}")
        path = os.path.realpath(os.path.join(base_dir, name))
        if os.path.commonpath([base_dir, path]) != base_dir:
            raise ValueError(f"视频路径越出 VIDEO_DIR: {name}")
        if not os.path.isfile(path):
            raise ValueError(f"视频文件不存在: {path}")
        paths.append(path)
    return paths


class MarioClient:
    def __init__(self, api_key: str, caller: str, base_url: str):
        self.base = base_url.rstrip("/")
        self.caller = caller
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {api_key}",
            "X-Caller-User-Id": caller,
            "Content-Type": "application/json",
        })

    def _check(self, r: requests.Response) -> dict:
        try:
            body = r.json()
        except Exception:
            r.raise_for_status()
            raise RuntimeError(f"非 JSON 响应: {r.status_code} {r.text[:300]}")
        if not body.get("success", False):
            raise RuntimeError(f"接口失败 [{r.status_code}]: {json.dumps(body, ensure_ascii=False)}")
        return body["data"]

    # ── Agent 配置 ───────────────────────────────────────────
    def update_system_prompt(self, agent_id: str, system_prompt: str) -> dict:
        """更新 Agent 的持久 system prompt；会影响后续调用。"""
        return self._check(self.s.patch(
            f"{self.base}/v1/agents/{agent_id}",
            json={"system": system_prompt},
        ))

    # ── 文件 ─────────────────────────────────────────────────
    def upload_file(self, file_path: str) -> str:
        """上传文件并返回 file_id。上传请求不能保留 application/json 头。"""
        headers = {
            "Authorization": self.s.headers["Authorization"],
            "X-Caller-User-Id": self.caller,
        }
        with open(file_path, "rb") as f:
            data = self._check(requests.post(
                f"{self.base}/v1/files",
                headers=headers,
                files={"file": (os.path.basename(file_path), f)},
            ))
        return data["id"]

    # ── Session ──────────────────────────────────────────────
    def create_session(self, agent_id: str, title: str = "OpenAPI 对话") -> str:
        data = self._check(self.s.post(
            f"{self.base}/v1/sessions",
            json={"agentId": agent_id, "title": title},
        ))
        return data["id"]

    # ── 发消息 ───────────────────────────────────────────────
    def send_message(self, session_id: str, text: str,
                     file_ids: Optional[List[str]] = None) -> dict:
        content = [{"type": "text", "text": text}]
        for file_id in file_ids or []:
            content.append({
                "type": "container_upload",
                "file_id": file_id,
            })
        return self._check(self.s.post(
            f"{self.base}/v1/sessions/{session_id}/events",
            json={"events": [
                {"type": "user.message", "content": content}
            ]},
        ))

    def interrupt(self, session_id: str) -> dict:
        return self._check(self.s.post(
            f"{self.base}/v1/sessions/{session_id}/events",
            json={"events": [{"type": "user.interrupt"}]},
        ))

    # ── 订阅 SSE，流式打印 agent 回复 ─────────────────────────
    def stream(self, session_id: str, after: int = 0, turn_id: Optional[str] = None):
        params = {"after": after}
        if turn_id:
            params["turn_id"] = turn_id
        with self.s.get(
            f"{self.base}/v1/sessions/{session_id}/events/stream",
            params=params,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=(10, 300),
        ) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[len("data:"):].strip()
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                yield evt


def run_once(client: MarioClient, agent_id: str, system_prompt: str,
             user_prompt: str, file_paths: Optional[List[str]] = None):
    # 文档只提供持久更新 system prompt 的 PATCH 接口，没有逐轮 system prompt 字段。
    client.update_system_prompt(agent_id, system_prompt)

    file_ids = []
    for file_path in file_paths or []:
        file_ids.append(client.upload_file(file_path))

    session_id = client.create_session(agent_id)
    print(f"[session] {session_id}", file=sys.stderr)

    resp = client.send_message(
        session_id,
        user_prompt,
        file_ids=file_ids,
    )
    turn_id = resp.get("turn_id")
    print(f"[turn] {turn_id}\n", file=sys.stderr)

    buf = []  # 累积 agent 正文
    for evt in client.stream(session_id, after=0, turn_id=turn_id):
        etype = evt.get("type", "")

        if etype == "agent.thinking":
            for c in evt.get("content", []):
                if c.get("type") == "thinking":
                    print(c.get("thinking", ""), end="", file=sys.stderr, flush=True)

        elif etype == "agent.message":
            for c in evt.get("content", []):
                if c.get("type") == "text":
                    buf.append(c.get("text", ""))

        elif etype == "agent.tool_use":
            print(f"[tool_use] {evt.get('name')} {json.dumps(evt.get('input', {}), ensure_ascii=False)}",
                  file=sys.stderr)

        elif etype == "agent.tool_result":
            print(f"[tool_result] {json.dumps(evt.get('content'), ensure_ascii=False)[:200]}",
                  file=sys.stderr)

        elif etype == "session.status_idle":
            print(f"[done] stop_reason={evt.get('stop_reason', {}).get('type')}", file=sys.stderr)
            break

        elif etype == "session.error":
            print(f"[error] {json.dumps(evt.get('error'), ensure_ascii=False)}", file=sys.stderr)
            break

    # 只输出纯 JSON：优先提取 ```json``` 代码块，否则截取第一个 { 到最后一个 }
    raw = "".join(buf).strip()
    print(extract_json(raw))
    return session_id


def extract_json(text: str) -> str:
    """从模型输出里剥离 markdown 代码围栏 / 前后文字，返回规整的 JSON 字符串。"""
    import re
    # 1) 优先取 ```json ... ``` 或 ``` ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = m.group(1).strip() if m else text.strip()
    # 2) 截取第一个 { 到最后一个 }（去掉围栏外残留文字）
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start:end + 1]
    # 3) 尝试解析并 re-dump，保证是合法且规整的 JSON；失败则原样返回
    try:
        return json.dumps(json.loads(candidate), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return candidate


def main():
    ap = argparse.ArgumentParser(description="Mario OpenAPI Agent 调用客户端")
    ap.add_argument("caller", help="调用者 username（邮箱前缀）或 kwaiUserId")
    ap.add_argument("--env", default="staging",
                    choices=list(BASE_URLS), help="环境，默认 staging")
    ap.add_argument("--base-url", default=None,
                    help="直接覆盖 base url")
    args = ap.parse_args()

    if not SYSTEM_PROMPT.strip():
        sys.exit("请先在 mario_client.py 顶部填写 SYSTEM_PROMPT")
    if not USER_PROMPT.strip():
        sys.exit("请先在 mario_client.py 顶部填写 USER_PROMPT")

    video_paths = load_selected_videos(VIDEO_DIR, VIDEO_SELECTION_FILE)
    base_url = args.base_url or BASE_URLS[args.env]
    client = MarioClient(API_KEY, args.caller, base_url)
    run_once(
        client,
        AGENT_ID,
        SYSTEM_PROMPT,
        USER_PROMPT,
        file_paths=video_paths,
    )


if __name__ == "__main__":
    main()

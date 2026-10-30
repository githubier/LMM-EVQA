"""Mario OpenAPI adapter used by interactive RSI rubric distillation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator

import requests

BASE_URLS = {
    "prod": "http://mario.internal/mario/openapi",
    "prt-idc": "http://mario-open-prt.internal/mario/openapi",
    "prt": "https://ee-mario.prt.kuaishou.com/mario/openapi",
    "staging": "https://kim-mario.staging.kuaishou.com/mario/openapi",
}


class MarioAPIError(RuntimeError):
    def __init__(self, status_code: int, error: object):
        self.status_code = status_code
        self.error = error
        super().__init__(
            f"Mario API failed [{status_code}]: "
            f"{json.dumps(error, ensure_ascii=False)}"
        )

    @property
    def error_type(self) -> str | None:
        return self.error.get("type") if isinstance(self.error, dict) else None


class MarioClient:
    def __init__(self, api_key: str, caller: str, base_url: str, timeout: int = 300):
        if not api_key:
            raise ValueError("Mario API key is empty")
        self.base = base_url.rstrip("/")
        self.caller = caller
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "X-Caller-User-Id": caller,
            "Content-Type": "application/json",
        })

    def _check(self, response: requests.Response) -> dict:
        try:
            body = response.json()
        except Exception as exc:
            response.raise_for_status()
            raise RuntimeError(f"Mario returned non-JSON: {response.text[:300]}") from exc
        if not body.get("success", False):
            raise MarioAPIError(response.status_code, body.get("error", body))
        return body["data"]

    def update_system_prompt(self, agent_id: str, prompt: str) -> None:
        self._check(self.session.patch(
            f"{self.base}/v1/agents/{agent_id}", json={"system": prompt},
            timeout=self.timeout,
        ))

    def preflight(self, agent_id: str, system_prompt: str | None = None) -> dict:
        """Verify Agent visibility without uploading files or starting inference."""
        result = {"agent_id": agent_id, "base_url": self.base}
        session_id = self.create_session(agent_id, "ECR RSI preflight")
        result["create_session"] = "ok"
        result["session_id"] = session_id
        if system_prompt is not None:
            self.update_system_prompt(agent_id, system_prompt)
            result["update_system_prompt"] = "ok"
        else:
            result["update_system_prompt"] = "skipped"
        return result

    def upload_file(self, path: str | Path) -> str:
        path = Path(path)
        headers = {
            "Authorization": self.session.headers["Authorization"],
            "X-Caller-User-Id": self.caller,
        }
        with path.open("rb") as stream:
            data = self._check(requests.post(
                f"{self.base}/v1/files", headers=headers,
                files={"file": (path.name, stream)}, timeout=self.timeout,
            ))
        return data["id"]

    def create_session(self, agent_id: str, title: str) -> str:
        return self._check(self.session.post(
            f"{self.base}/v1/sessions",
            json={"agentId": agent_id, "title": title}, timeout=self.timeout,
        ))["id"]

    def send_message(self, session_id: str, text: str, file_ids: list[str]) -> str | None:
        content = [{"type": "text", "text": text}]
        content.extend({"type": "container_upload", "file_id": item} for item in file_ids)
        data = self._check(self.session.post(
            f"{self.base}/v1/sessions/{session_id}/events",
            json={"events": [{"type": "user.message", "content": content}]},
            timeout=self.timeout,
        ))
        return data.get("turn_id")

    def interrupt(self, session_id: str) -> None:
        self._check(self.session.post(
            f"{self.base}/v1/sessions/{session_id}/events",
            json={"events": [{"type": "user.interrupt"}]},
            timeout=self.timeout,
        ))

    def send_completion_message(self, session_id: str, text: str) -> tuple[str | None, bool]:
        """Send a no-file follow-up, interrupting a stale running turn if required."""
        interrupted = False
        try:
            return self.send_message(session_id, text, []), interrupted
        except MarioAPIError as exc:
            if exc.status_code != 409 or exc.error_type != "session_busy":
                raise
            self.interrupt(session_id)
            interrupted = True
            time.sleep(1)
            return self.send_message(session_id, text, []), interrupted

    def stream(self, session_id: str, turn_id: str | None) -> Iterator[dict]:
        params = {"after": 0}
        if turn_id:
            params["turn_id"] = turn_id
        with self.session.get(
            f"{self.base}/v1/sessions/{session_id}/events/stream",
            params=params, headers={"Accept": "text/event-stream"},
            stream=True, timeout=(10, self.timeout),
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                try:
                    yield json.loads(raw[len("data:"):].strip())
                except json.JSONDecodeError:
                    continue

    def _collect_turn(self, session_id: str, turn_id: str | None) -> tuple[str, str | None]:
        chunks = []
        stop_reason = None
        for event in self.stream(session_id, turn_id):
            kind = event.get("type")
            if kind == "agent.message":
                chunks.extend(
                    item.get("text", "") for item in event.get("content", [])
                    if item.get("type") == "text"
                )
            elif kind == "session.error":
                raise RuntimeError(f"Mario session failed: {event.get('error')}")
            elif kind == "session.status_idle":
                stop_reason = event.get("stop_reason", {}).get("type")
                break
        return "".join(chunks).strip(), stop_reason

    def run(self, agent_id: str, system_prompt: str, user_prompt: str,
            files: list[str | Path], title: str,
            completion_retries: int = 2) -> tuple[dict, dict]:
        # Mario exposes a persistent system prompt update, not a per-turn system field.
        self.update_system_prompt(agent_id, system_prompt)
        file_ids = [self.upload_file(path) for path in files]
        session_id = self.create_session(agent_id, title)
        turn_id = self.send_message(session_id, user_prompt, file_ids)
        turns = []
        completion_prompt = (
            "继续完成刚才的任务。不要再描述分析过程；请现在直接输出任务要求的完整、"
            "合法 JSON 对象，不能输出 Markdown、前言、后记或中间状态。"
        )
        for attempt in range(completion_retries + 1):
            text, stop_reason = self._collect_turn(session_id, turn_id)
            turns.append({
                "turn_id": turn_id,
                "stop_reason": stop_reason,
                "response_preview": text[:1000],
            })
            try:
                result = extract_json_object(text)
                return result, {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "stop_reason": stop_reason,
                    "uploaded_file_count": len(file_ids),
                    "completion_followups": attempt,
                    "turns": turns,
                }
            except ValueError:
                if attempt >= completion_retries:
                    raise ValueError(
                        "Mario produced no JSON after "
                        f"{completion_retries} completion follow-ups; "
                        f"session_id={session_id}; responses={turns}"
                    )
                turn_id, interrupted = self.send_completion_message(
                    session_id, completion_prompt
                )
                turns[-1]["interrupted_stale_turn"] = interrupted
        raise AssertionError("unreachable")


def extract_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"Mario response contains no JSON object: {text[:500]!r}")


def api_key_from_env(name: str = "MARIO_API_KEY") -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"set {name}; never put API keys in source or command history")
    return value

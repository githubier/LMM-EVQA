#!/usr/bin/env python3
"""Check Mario credentials, Agent visibility and prompt-update permission."""

from __future__ import annotations

import argparse
import json
import os

from rsi.distill import SYSTEM_PROMPT
from rsi.mario import BASE_URLS, MarioClient, api_key_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caller", required=True)
    parser.add_argument("--agent-id", default=os.environ.get("MARIO_AGENT_ID"))
    parser.add_argument("--env", choices=sorted(BASE_URLS), default="prt-idc")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--skip-prompt-update", action="store_true",
        help="Only verify that a session can be created for this Agent.",
    )
    args = parser.parse_args()
    if not args.agent_id:
        parser.error("set --agent-id or MARIO_AGENT_ID")
    client = MarioClient(
        api_key_from_env(), args.caller, args.base_url or BASE_URLS[args.env]
    )
    try:
        result = client.preflight(
            args.agent_id,
            None if args.skip_prompt_update else SYSTEM_PROMPT,
        )
    except Exception as exc:
        raise SystemExit(
            "Mario preflight failed before any video upload. "
            "Check that MARIO_AGENT_ID belongs to the selected --env and is visible "
            f"to --caller. Cause: {type(exc).__name__}: {exc}"
        ) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

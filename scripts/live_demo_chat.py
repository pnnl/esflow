"""Drive the live planner chat over /api/chat and print the streamed text.

Temporary manual verification harness for the onboarding demonstration; not a
pytest test (it costs real model calls). Usage:

    python scripts/live_demo_chat.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app import app


TURNS = [
    "I have my own Python function for an analysis your tools don't cover. "
    "What format does my code need to be in, and where do I put it?",
    "Please show me the demonstration running end to end.",
]


def _payload(history: list[dict], text: str) -> dict:
    # Every message needs its own id: the Vercel AI request schema requires it.
    history.append(
        {
            "id": f"m{len(history)}",
            "role": "user",
            "parts": [{"type": "text", "text": text}],
        }
    )
    return {"id": "live", "messages": history, "trigger": "submit-message"}


async def main() -> None:
    history: list[dict] = []
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=600
    ) as client:
        for turn in TURNS:
            print("=" * 78)
            print("USER:", turn)
            print("-" * 78)
            reply: list[str] = []
            async with client.stream(
                "POST", "/api/chat", json=_payload(history, turn)
            ) as response:
                print("HTTP", response.status_code)
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    body = line[len("data: ") :]
                    if body == "[DONE]":
                        continue
                    try:
                        event = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    kind = event.get("type", "")
                    if kind == "text-delta":
                        reply.append(event.get("delta", ""))
                    elif kind == "tool-input-available":
                        print(f"\n[tool call] {event.get('toolName')}")
                    elif kind == "tool-output-available":
                        out = str(event.get("output", ""))
                        print(f"[tool result] {out[:600]}")
            text = "".join(reply)
            print("\nASSISTANT:", text)
            history.append(
                {
                    "id": f"m{len(history)}",
                    "role": "assistant",
                    "parts": [{"type": "text", "text": text}],
                }
            )


if __name__ == "__main__":
    asyncio.run(main())

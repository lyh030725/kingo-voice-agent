"""Agentic visualization gate for realtime teaching turns."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass

from brain import xai_client

log = logging.getLogger("visual-router")

VISUAL_ROUTER_MODEL = os.environ.get(
    "VISUAL_ROUTER_MODEL",
    os.environ.get("CHAT_MODEL", "grok-4.3"),
)
VISUAL_ROUTER_TIMEOUT_S = float(os.environ.get("VISUAL_ROUTER_TIMEOUT_S", "3.0"))

VISUAL_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "decide_visual",
        "description": (
            "Decide whether a visual materially helps the current teaching step and, "
            "when it does, return render-ready visual data."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "needed": {"type": "boolean"},
                "kind": {
                    "type": "string",
                    "enum": ["none", "formula", "flow", "pdf"],
                },
                "title": {"type": "string"},
                "caption": {"type": "string"},
                "latex": {"type": "string"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 8,
                },
                "file": {"type": "string"},
                "page": {"type": "integer", "minimum": 0},
            },
            "required": [
                "needed",
                "kind",
                "title",
                "caption",
                "latex",
                "labels",
                "file",
                "page",
            ],
            "additionalProperties": False,
        },
    },
}

VISUAL_DECISION_PROMPT = """
Decide whether the current teaching step benefits materially from a visual.
Consider the recent conversation, not only the latest utterance, so short follow-ups
like "네" or "설명해 줘요" inherit the active concept.

Use formula for an equation or variable relationship, flow for a process/structure,
and pdf only when an exact course filename and page are already present in context.
Prefer a useful visual when it makes the concept easier to reason about; do not use
one merely for decoration. Keep the caption as a clue, not the final answer.

If no visual is useful, set needed=false, kind=none, and leave the visual fields empty
(labels=[] and page=0). Never invent a PDF filename or page.
""".strip()


@dataclass(frozen=True)
class VisualDecision:
    needed: bool
    kind: str = "none"
    args: dict | None = None


def _decision_request(current_user: str, recent_conversation: list[dict]) -> dict:
    context = {
        "recent_conversation": recent_conversation[-6:],
        "current_user": current_user.strip(),
    }
    return {
        "model": VISUAL_ROUTER_MODEL,
        "reasoning_effort": "none",
        "max_completion_tokens": 500,
        "messages": [
            {"role": "system", "content": VISUAL_DECISION_PROMPT},
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "tools": [VISUAL_DECISION_TOOL],
        "tool_choice": {
            "type": "function",
            "function": {"name": "decide_visual"},
        },
        "parallel_tool_calls": False,
    }


def _parse_decision(message) -> VisualDecision:
    calls = message.tool_calls or []
    if not calls:
        return VisualDecision(False)
    payload = json.loads(calls[0].function.arguments or "{}")
    needed = bool(payload.get("needed"))
    kind = str(payload.get("kind", "none"))
    if not needed or kind == "none":
        return VisualDecision(False)
    if kind not in {"formula", "flow", "pdf"}:
        return VisualDecision(False)

    common = {
        "kind": kind,
        "title": str(payload.get("title", "")).strip(),
        "caption": str(payload.get("caption", "")).strip(),
    }
    if not common["title"] or not common["caption"]:
        return VisualDecision(False)
    if kind == "formula":
        latex = str(payload.get("latex", "")).strip()
        if not latex:
            return VisualDecision(False)
        common["latex"] = latex
    elif kind == "flow":
        labels = [
            str(label).strip()
            for label in payload.get("labels", [])
            if str(label).strip()
        ]
        if len(labels) < 2:
            return VisualDecision(False)
        common["labels"] = labels[:8]
    else:
        file = str(payload.get("file", "")).strip()
        page = int(payload.get("page", 0) or 0)
        if not file or page < 1:
            return VisualDecision(False)
        common["file"] = file
        common["page"] = page
    return VisualDecision(True, kind, common)


async def decide_visualization(
    current_user: str,
    recent_conversation: list[dict],
) -> VisualDecision:
    """Force Grok to make a structured visual/no-visual decision for one turn."""
    if not current_user.strip():
        return VisualDecision(False)

    request = _decision_request(current_user, recent_conversation)

    def complete():
        response = xai_client().chat.completions.create(**request)
        return response.choices[0].message

    try:
        message = await asyncio.wait_for(
            asyncio.to_thread(complete),
            VISUAL_ROUTER_TIMEOUT_S,
        )
        decision = _parse_decision(message)
        log.info(
            "visual decision needed=%s kind=%s user=%r",
            decision.needed,
            decision.kind,
            current_user[:120],
        )
        return decision
    except TimeoutError:
        log.warning("visual decision timed out")
    except Exception:
        log.exception("visual decision failed")
    return VisualDecision(False)

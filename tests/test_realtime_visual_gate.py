from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["VOICE_AI_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_spec
import grok_live
from grok_live import GrokTransport
from transport import AgentAudio, AgentTextDelta, AgentTurnDone, ToolCalled
from visual_router import VisualDecision


class FakeSocket:
    def __init__(self, events: list[dict]) -> None:
        self.events = iter(json.dumps(event) for event in events)
        self.sent: list[dict] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


class RealtimeVisualGateTests(unittest.TestCase):
    def test_audio_turn_cancels_auto_response_then_shows_visual_before_real_response(self) -> None:
        async def scenario() -> None:
            transport = GrokTransport()
            transport._ws = FakeSocket([
                {"type": "input_audio_buffer.speech_started"},
                {"type": "input_audio_buffer.speech_stopped"},
                {"type": "response.created"},
                {
                    "type": "response.output_audio.delta",
                    "delta": base64.b64encode(b"discard-me").decode(),
                },
                {
                    "type": "response.output_audio_transcript.delta",
                    "delta": "수식을 그냥 읽을게요.",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "네, 설명해 줘요.",
                },
                {"type": "response.done"},
                {"type": "response.created"},
                {
                    "type": "response.output_audio_transcript.delta",
                    "delta": "그림에서 스케일링 부분을 먼저 볼까요?",
                },
                {
                    "type": "response.output_audio_transcript.done",
                    "transcript": "그림에서 스케일링 부분을 먼저 볼까요?",
                },
                {"type": "response.done"},
            ])
            transport._connected.set()

            decision = VisualDecision(
                True,
                "formula",
                {
                    "kind": "formula",
                    "title": "스케일드 닷 프로덕트 어텐션",
                    "caption": "점곱을 차원 크기로 스케일링해요.",
                    "latex": r"\operatorname{softmax}(QK^T/\sqrt{d_k})V",
                },
            )
            visual = {
                "kind": "formula",
                "title": "스케일드 닷 프로덕트 어텐션",
                "caption": "점곱을 차원 크기로 스케일링해요.",
                "latex": r"\operatorname{softmax}(QK^T/\sqrt{d_k})V",
            }

            with (
                patch.object(
                    grok_live,
                    "decide_visualization",
                    new=AsyncMock(return_value=decision),
                ),
                patch.object(
                    agent_spec,
                    "run_tool",
                    new=AsyncMock(return_value=visual),
                ),
                patch.object(transport, "_schedule_memory_refresh"),
                patch.object(grok_live.EXTERNAL_BRAIN, "schedule"),
            ):
                events = [event async for event in transport.events()]

            self.assertFalse(any(isinstance(event, AgentAudio) for event in events))
            self.assertEqual(
                [event.text for event in events if isinstance(event, AgentTextDelta)],
                ["그림에서 스케일링 부분을 먼저 볼까요?"],
            )
            visuals = [
                event for event in events
                if isinstance(event, ToolCalled) and event.name == "show_visualization"
            ]
            self.assertEqual(len(visuals), 1)
            self.assertEqual(visuals[0].result, visual)
            self.assertEqual(sum(isinstance(event, AgentTurnDone) for event in events), 1)

            sent = transport._ws.sent
            self.assertIn({"type": "response.cancel"}, sent)
            routed = [item for item in sent if item.get("type") == "response.create"]
            self.assertEqual(len(routed), 1)
            self.assertIn("visual has already been shown", routed[0]["response"]["instructions"])

        asyncio.run(scenario())

    def test_typed_turn_routes_visual_before_response_create(self) -> None:
        async def scenario() -> None:
            transport = GrokTransport()
            transport._ws = FakeSocket([])
            tool_event = ToolCalled(
                "show_visualization",
                {"kind": "flow"},
                {"kind": "flow", "title": "흐름", "caption": "단계", "labels": ["A", "B"]},
            )

            with (
                patch.object(transport, "_refresh_memory_context", new=AsyncMock()),
                patch.object(
                    transport,
                    "_prepare_visual",
                    new=AsyncMock(return_value=(tool_event, "visual instruction")),
                ),
            ):
                await transport.send_text("과정을 설명해 줘")

            self.assertEqual(transport._local_events.qsize(), 1)
            sent = transport._ws.sent
            self.assertEqual(sent[0]["type"], "conversation.item.create")
            self.assertEqual(sent[1], {
                "type": "response.create",
                "response": {"instructions": "visual instruction"},
            })

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

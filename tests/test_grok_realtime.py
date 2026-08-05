from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["VOICE_AI_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_spec
import server
from grok_live import GrokTransport
from transport import AgentTextDelta, AgentTurnDone, ToolCalled, UserStartedSpeaking


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


class GrokRealtimeTests(unittest.TestCase):
    def test_barge_in_flush_signal_also_cancels_model_response(self) -> None:
        async def scenario() -> None:
            transport = GrokTransport()
            transport._ws = FakeSocket([
                {"type": "input_audio_buffer.speech_started"},
                {"type": "response.done"},
            ])
            transport._connected.set()
            transport._agent_speaking = True

            events = [event async for event in transport.events()]

            self.assertIsInstance(events[0], UserStartedSpeaking)
            self.assertIsInstance(events[1], AgentTurnDone)
            self.assertIn({"type": "response.cancel"}, transport._ws.sent)

        asyncio.run(scenario())

    def test_multiple_tools_trigger_one_follow_up_response(self) -> None:
        async def scenario() -> None:
            transport = GrokTransport()
            transport._ws = FakeSocket([
                {"type": "response.function_call_arguments.done", "name": "one", "call_id": "1", "arguments": "{}"},
                {"type": "response.function_call_arguments.done", "name": "two", "call_id": "2", "arguments": "{}"},
                {"type": "response.done"},
            ])
            transport._connected.set()

            with patch.object(agent_spec, "run_tool", AsyncMock(return_value={"ok": True})):
                events = [event async for event in transport.events()]

            self.assertEqual(sum(isinstance(event, ToolCalled) for event in events), 2)
            self.assertEqual(transport._ws.sent.count({"type": "response.create"}), 1)

        asyncio.run(scenario())

    def test_voice_transcript_delta_is_forwarded_immediately(self) -> None:
        async def scenario() -> None:
            transport = GrokTransport()
            transport._ws = FakeSocket([
                {"type": "response.output_audio_transcript.delta", "delta": "바로"},
            ])
            transport._connected.set()

            events = [event async for event in transport.events()]

            self.assertEqual(events, [AgentTextDelta("바로")])

        asyncio.run(scenario())

    def test_browser_keeps_mic_open_and_flushes_pcm_queue(self) -> None:
        page = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")

        self.assertNotIn('state === "speaking"', page)
        self.assertIn('message.type === "flush"', page)
        self.assertIn('message.type === "token"', page)
        self.assertIn("voicePending.bubble.append(message.text)", page)
        self.assertIn("function flushPlayback()", page)
        self.assertIn("for (const source of activeSources)", page)

    def test_voice_tool_result_reaches_visualization_renderer(self) -> None:
        visualization = {
            "title": "변화량",
            "kind": "plot",
            "caption": "증가 추세예요.",
            "latex": "",
            "labels": [],
            "points": [{"x": 0, "y": 1}, {"x": 1, "y": 2}],
            "x_label": "시간",
            "y_label": "값",
        }

        class Browser:
            def __init__(self) -> None:
                self.sent = []

            async def send_json(self, message: dict) -> None:
                self.sent.append(message)

        class Provider:
            async def events(self):
                yield ToolCalled("show_visualization", {}, visualization)

        browser = Browser()
        asyncio.run(server.pump_provider_events(browser, Provider()))

        self.assertIn({"type": "visualization", "visualization": visualization}, browser.sent)


if __name__ == "__main__":
    unittest.main()

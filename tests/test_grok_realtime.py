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
from transport import AgentAudio, AgentTextDelta, AgentTurnDone, Failed, ToolCalled, Transcript, UserStartedSpeaking


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
    def test_filler_instruction_is_voice_only(self) -> None:
        self.assertIn("Immediately before the first tool call", agent_spec.persona("socratic"))
        self.assertNotIn("Voice-only filler", agent_spec.SYSTEM_PROMPT)

    def test_barge_in_cancels_active_response_and_discards_remaining_audio(self) -> None:
        async def scenario() -> None:
            transport = GrokTransport()
            transport._ws = FakeSocket([
                {"type": "response.created"},
                {"type": "response.output_audio.delta", "delta": "AQI="},
                {"type": "input_audio_buffer.speech_started"},
                {"type": "response.output_audio.delta", "delta": "AwQ="},
                {"type": "error", "error": {"message": "Cancellation failed: no active response found"}},
                {"type": "response.done"},
            ])
            transport._connected.set()

            events = [event async for event in transport.events()]

            self.assertIn(UserStartedSpeaking(), events)
            self.assertIn(AgentTurnDone(), events)
            self.assertEqual(sum(isinstance(event, AgentAudio) for event in events), 1)
            self.assertFalse(any(isinstance(event, Failed) for event in events))
            self.assertIn({"type": "response.cancel"}, transport._ws.sent)

        asyncio.run(scenario())

    def test_realtime_errors_other_than_stale_cancel_are_forwarded(self) -> None:
        async def scenario() -> None:
            transport = GrokTransport()
            transport._ws = FakeSocket([
                {"type": "error", "error": {"message": "authentication expired"}},
            ])
            transport._connected.set()

            events = [event async for event in transport.events()]

            self.assertEqual(events, [Failed("authentication expired")])

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

    def test_filler_and_final_answer_deltas_are_forwarded(self) -> None:
        async def scenario() -> None:
            transport = GrokTransport()
            transport._ws = FakeSocket([
                {"type": "response.output_audio_transcript.delta", "delta": "잠시만요."},
                {"type": "response.output_audio_transcript.done", "transcript": "잠시만요."},
                {"type": "response.function_call_arguments.done", "name": "search", "call_id": "1", "arguments": "{}"},
                {"type": "response.done"},
                {"type": "response.output_audio_transcript.delta", "delta": "찾았어요."},
                {"type": "response.output_audio_transcript.done", "transcript": "찾았어요."},
                {"type": "response.done"},
            ])
            transport._connected.set()

            with patch.object(agent_spec, "run_tool", AsyncMock(return_value={"ok": True})):
                events = [event async for event in transport.events()]

            class Browser:
                def __init__(self) -> None:
                    self.sent = []

                async def send_json(self, message: dict) -> None:
                    self.sent.append(message)

            class Provider:
                async def events(self):
                    for event in events:
                        yield event

            browser = Browser()
            await server.pump_provider_events(browser, Provider())

            self.assertEqual(
                [event.text for event in events if isinstance(event, AgentTextDelta)],
                ["잠시만요.", "찾았어요."],
            )
            self.assertNotIn(Transcript("agent", "잠시만요."), events)
            self.assertIn(Transcript("agent", "찾았어요."), events)
            self.assertEqual(transport._ws.sent.count({"type": "response.create"}), 1)
            self.assertEqual(
                [message for message in browser.sent if message["type"] == "token"],
                [
                    {"type": "token", "text": "잠시만요."},
                    {"type": "token", "text": "찾았어요."},
                ],
            )
            self.assertEqual(
                [message for message in browser.sent if message["type"] == "transcript"],
                [{"type": "transcript", "who": "agent", "text": "찾았어요."}],
            )

        asyncio.run(scenario())

    def test_cumulative_user_transcript_updates_ignore_exact_duplicates(self) -> None:
        async def scenario() -> None:
            transport = GrokTransport()
            transport._ws = FakeSocket([
                {"type": "conversation.item.input_audio_transcription.updated", "transcript": "어, 소프트"},
                {"type": "conversation.item.input_audio_transcription.updated", "transcript": "어, 소프트맥스 연산이 뭐야?"},
                {"type": "conversation.item.input_audio_transcription.updated", "transcript": "어, 소프트맥스 연산이 뭐야?"},
            ])
            transport._connected.set()

            events = [event async for event in transport.events()]

            self.assertEqual(events, [
                Transcript("user", "어, 소프트"),
                Transcript("user", "어, 소프트맥스 연산이 뭐야?"),
            ])

        asyncio.run(scenario())

    def test_agent_transcript_streams_deltas_and_publishes_final_text(self) -> None:
        async def scenario() -> None:
            transport = GrokTransport()
            transport._ws = FakeSocket([
                {"type": "response.created"},
                {"type": "response.output_audio_transcript.delta", "delta": "소프트맥스는 "},
                {"type": "response.output_audio_transcript.delta", "delta": "확률로 바꿉니다."},
                {"type": "response.output_audio_transcript.done", "transcript": "소프트맥스는 확률로 바꿉니다."},
                {"type": "response.done"},
                {"type": "response.output_audio_transcript.done", "transcript": "소프트맥스는 확률로 바꿉니다."},
            ])
            transport._connected.set()

            events = [event async for event in transport.events()]

            self.assertEqual(
                [event.text for event in events if isinstance(event, AgentTextDelta)],
                ["소프트맥스는 ", "확률로 바꿉니다."],
            )
            self.assertEqual(
                [event for event in events if isinstance(event, Transcript)],
                [Transcript("agent", "소프트맥스는 확률로 바꿉니다.")],
            )
            self.assertEqual(sum(isinstance(event, AgentTurnDone) for event in events), 1)

        asyncio.run(scenario())

    def test_browser_keeps_mic_open_and_flushes_pcm_queue(self) -> None:
        page = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")

        self.assertNotIn('state === "speaking"', page)
        self.assertIn('message.type === "flush"', page)
        self.assertIn('message.type === "token"', page)
        self.assertIn("voicePending.bubble.append(message.text)", page)
        self.assertIn("function flushPlayback()", page)
        self.assertIn("for (const source of activeSources)", page)

    def test_browser_coalesces_voice_transcripts_and_waits_for_playback(self) -> None:
        page = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn("voiceUserPending.bubble.textContent = message.text", page)
        self.assertIn("function replaceVoiceAssistantMessage", page)
        self.assertIn("voicePending || voiceLastAssistant", page)
        self.assertIn("voiceLastAssistant = voicePending", page)
        self.assertIn(
            '} else if (message.type === "tool") {\n'
            "    if (voicePending) {\n"
            '      setSymbolState(voicePending, "sustain");\n'
            "      voicePending = null;",
            page,
        )
        self.assertNotIn("voicePending.row.remove()", page)
        self.assertIn('message.type === "turn_done"', page)
        self.assertIn("activeSources.size === 0", page)
        self.assertIn("@keyframes voice-heartbeat", page)
        self.assertIn("prefers-reduced-motion: reduce", page)

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

    def test_agent_turn_done_waits_for_browser_playback(self) -> None:
        class Browser:
            def __init__(self) -> None:
                self.sent = []

            async def send_json(self, message: dict) -> None:
                self.sent.append(message)

        class Provider:
            async def events(self):
                yield AgentTurnDone()

        browser = Browser()
        asyncio.run(server.pump_provider_events(browser, Provider()))

        self.assertEqual(browser.sent, [{"type": "turn_done"}])

    def test_barge_in_flushes_playback_after_provider_turn_is_done(self) -> None:
        class Browser:
            def __init__(self) -> None:
                self.sent = []

            async def send_json(self, message: dict) -> None:
                self.sent.append(message)

        class Provider:
            async def events(self):
                yield AgentTurnDone()
                yield UserStartedSpeaking()

        browser = Browser()
        asyncio.run(server.pump_provider_events(browser, Provider()))

        self.assertEqual(browser.sent, [
            {"type": "turn_done"},
            {"type": "state", "value": "hearing"},
            {"type": "flush"},
        ])


if __name__ == "__main__":
    unittest.main()

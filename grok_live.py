"""xAI realtime Speech-to-Speech transport, adapted from course Week 4."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import AsyncIterator

import websockets

import agent_spec
from transport import (
    AGENT_RATE,
    CALLER_RATE,
    AgentAudio,
    AgentTurnDone,
    Failed,
    SessionReady,
    ToolCalled,
    Transcript,
    Transport,
    UserStartedSpeaking,
    UserStoppedSpeaking,
)

log = logging.getLogger("grok-live")

MODEL = os.environ.get("GROK_VOICE_MODEL", "grok-voice-latest")
VOICE = os.environ.get("GROK_VOICE") or os.environ.get("TTS_VOICE") or "eve"
WS_URL = os.environ.get("XAI_REALTIME_URL", "wss://api.x.ai/v1/realtime")
READY_TIMEOUT_S = 10


class GrokTransport(Transport):
    name = "grok"

    def __init__(self, mode: str = "socratic") -> None:
        self.mode = mode
        self._ws = None
        self._connected = asyncio.Event()
        self._ready = asyncio.Event()
        self._closed = False
        self._tools_this_turn = 0
        self._agent_speaking = False
        self._response_transcript = ""

    async def start(self) -> None:
        key = os.environ.get("XAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("XAI_API_KEY is not set")

        self._ws = await websockets.connect(
            f"{WS_URL}?model={MODEL}",
            additional_headers={"Authorization": f"Bearer {key}"},
            max_size=None,
            ping_interval=20,
        )
        self._connected.set()
        await self._send({
            "type": "session.update",
            "session": {
                "instructions": agent_spec.persona(self.mode),
                "voice": VOICE,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": CALLER_RATE},
                        "transcription": {"model": "grok-transcribe", "language_hint": "ko"},
                    },
                    "output": {"format": {"type": "audio/pcm", "rate": AGENT_RATE}},
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": float(os.environ.get("VAD_THRESHOLD", "0.5")),
                    "prefix_padding_ms": int(os.environ.get("PREFIX_MS", "300")),
                    "silence_duration_ms": int(os.environ.get("SILENCE_MS", "700")),
                },
                "tools": agent_spec.json_schemas(),
            },
        })
        try:
            await asyncio.wait_for(self._ready.wait(), READY_TIMEOUT_S)
        except TimeoutError as exc:
            raise RuntimeError("xAI realtime session configuration timed out") from exc

    async def send_audio(self, pcm: bytes) -> None:
        if self._ws is not None and not self._closed:
            await self._send({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            })

    async def events(self) -> AsyncIterator:
        await self._connected.wait()
        assert self._ws is not None
        try:
            async for raw in self._ws:
                event = json.loads(raw)
                kind = event.get("type", "")

                if kind == "session.updated":
                    self._ready.set()
                    yield SessionReady()
                elif kind == "input_audio_buffer.speech_started":
                    yield UserStartedSpeaking()
                    if self._agent_speaking:
                        self._agent_speaking = False
                        await self._send({"type": "response.cancel"})
                elif kind == "input_audio_buffer.speech_stopped":
                    yield UserStoppedSpeaking()
                elif kind == "response.output_audio.delta":
                    self._agent_speaking = True
                    yield AgentAudio(base64.b64decode(event["delta"]))
                elif kind == "response.output_audio_transcript.delta":
                    self._response_transcript += event.get("delta", "")
                elif kind == "response.function_call_arguments.done":
                    name = event.get("name", "")
                    call_id = event.get("call_id", "")
                    try:
                        args = json.loads(event.get("arguments") or "{}")
                        result = await agent_spec.run_tool(name, args)
                    except (json.JSONDecodeError, TypeError) as exc:
                        args, result = {}, {"error": str(exc)}
                    await self._send({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(result, ensure_ascii=False),
                        },
                    })
                    self._tools_this_turn += 1
                    yield ToolCalled(name, args, result)
                elif kind == "response.done":
                    if self._tools_this_turn:
                        self._tools_this_turn = 0
                        self._response_transcript = ""
                        await self._send({"type": "response.create"})
                    else:
                        self._agent_speaking = False
                        if self._response_transcript:
                            yield Transcript("agent", self._response_transcript)
                        self._response_transcript = ""
                        yield AgentTurnDone()
                elif kind.endswith("input_audio_transcription.completed"):
                    yield Transcript("user", event.get("transcript", ""))
                elif kind.endswith("output_audio_transcript.done"):
                    self._response_transcript = event.get("transcript") or self._response_transcript
                elif kind == "error":
                    yield Failed(event.get("error", {}).get("message", json.dumps(event)))
        except websockets.ConnectionClosed as exc:
            yield Failed(f"model connection closed: {exc}")
        finally:
            self._closed = True

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def _send(self, payload: dict) -> None:
        assert self._ws is not None
        await self._ws.send(json.dumps(payload))

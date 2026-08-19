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
from brain_runtime import (
    MAX_HISTORY_MESSAGES,
    bootstrap_memory_context,
    get_external_brain,
    prefetch_memory_context,
)
from transport import (
    AGENT_RATE,
    CALLER_RATE,
    AgentAudio,
    AgentTextBoundary,
    AgentTextDelta,
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
TOOL_TIMEOUT_S = 10
MEMORY_BOOTSTRAP_TIMEOUT_S = float(os.environ.get("MEMORY_BOOTSTRAP_TIMEOUT_S", "2.5"))

# Compatibility for legacy tests/default single-student callers.
EXTERNAL_BRAIN = get_external_brain("default-student")


class GrokTransport(Transport):
    name = "grok"

    def __init__(
        self,
        mode: str = "socratic",
        *,
        student_id: str = "default-student",
        session_id: str = "default-session",
    ) -> None:
        self.mode = mode
        self.student_id = student_id
        self.session_id = session_id
        self._ws = None
        self._connected = asyncio.Event()
        self._ready = asyncio.Event()
        self._closed = False
        self._tools_this_turn = 0
        self._response_transcript = ""
        self._response_transcript_emitted = False
        self._response_done = False
        self._last_user_transcript = ""
        self._response_active = False
        self._discard_response_output = False
        self._conversation: list[dict[str, str]] = []
        self._assessment_scheduled = False
        self._memory_context: dict = {"found": False}
        self._memory_task: asyncio.Task | None = None
        self._memory_query = ""

    async def start(self) -> None:
        key = os.environ.get("XAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("XAI_API_KEY is not set")

        # Load a small recent weak-concept snapshot before the voice session is
        # configured. Turn-specific semantic recall can refine this later, but
        # the very first response already has personalized learner state.
        try:
            self._memory_context = await asyncio.wait_for(
                bootstrap_memory_context(self.student_id),
                MEMORY_BOOTSTRAP_TIMEOUT_S,
            )
            log.info(
                "realtime memory bootstrap student=%s found=%s memories=%s",
                self.student_id,
                bool(self._memory_context.get("found")),
                len(self._memory_context.get("memories", [])),
            )
        except TimeoutError:
            log.warning("realtime memory bootstrap timed out student=%s", self.student_id)
            self._memory_context = {"found": False}
        except Exception:
            log.exception("realtime memory bootstrap failed student=%s", self.student_id)
            self._memory_context = {"found": False}

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
                "instructions": agent_spec.persona(self.mode, self._memory_context),
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
            await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})

    async def send_text(self, text: str) -> None:
        if self._ws is not None and not self._closed:
            text = text.strip()
            self._begin_user_turn(text)
            await self._refresh_memory_context(text)
            await self._send({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            })
            await self._send({"type": "response.create"})

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
                elif kind == "response.created":
                    self._response_transcript = ""
                    self._response_transcript_emitted = False
                    self._response_done = False
                    self._response_active = True
                    self._discard_response_output = False
                elif kind == "input_audio_buffer.speech_started":
                    yield UserStartedSpeaking()
                    self._reset_for_new_speech()
                    if self._response_active:
                        self._response_active = False
                        self._discard_response_output = True
                        await self._send({"type": "response.cancel"})
                elif kind == "input_audio_buffer.speech_stopped":
                    yield UserStoppedSpeaking()
                elif kind == "response.output_audio.delta":
                    if not self._discard_response_output:
                        yield AgentAudio(base64.b64decode(event["delta"]))
                elif kind == "response.output_audio_transcript.delta":
                    if self._discard_response_output:
                        continue
                    delta = event.get("delta", "")
                    self._response_transcript += delta
                    if delta:
                        yield AgentTextDelta(delta)
                elif kind == "response.function_call_arguments.done":
                    name = event.get("name", "")
                    call_id = event.get("call_id", "")
                    try:
                        args = json.loads(event.get("arguments") or "{}")
                        result = await asyncio.wait_for(agent_spec.run_tool(name, args), TOOL_TIMEOUT_S)
                    except TimeoutError:
                        result = {"error": f"{name} timed out"}
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
                    self._response_active = False
                    if self._discard_response_output:
                        self._tools_this_turn = 0
                        self._discard_response_output = False
                        self._response_transcript = ""
                        self._response_transcript_emitted = False
                        self._response_done = False
                        yield AgentTurnDone()
                    elif self._tools_this_turn:
                        self._tools_this_turn = 0
                        had_interim_text = bool(self._response_transcript)
                        self._response_transcript = ""
                        self._response_transcript_emitted = False
                        self._response_done = False
                        await self._send({"type": "response.create"})
                        if had_interim_text:
                            yield AgentTextBoundary()
                    else:
                        self._response_done = True
                        self._response_transcript = self._response_transcript or _transcript_from_response(event)
                        if self._response_transcript and not self._response_transcript_emitted:
                            yield Transcript("agent", self._response_transcript)
                            self._response_transcript_emitted = True
                        self._schedule_assessment()
                        yield AgentTurnDone()
                elif kind in {
                    "conversation.item.input_audio_transcription.updated",
                    "conversation.item.input_audio_transcription.completed",
                } or kind.endswith("input_audio_transcription.completed"):
                    transcript = event.get("transcript") or event.get("text") or ""
                    if transcript and transcript != self._last_user_transcript:
                        self._last_user_transcript = transcript
                        self._schedule_memory_prefetch(transcript)
                        yield Transcript("user", transcript)
                    if kind.endswith("input_audio_transcription.completed") and transcript:
                        self._begin_user_turn(transcript, keep_memory_task=True)
                        self._schedule_memory_prefetch(transcript)
                        if self._response_done:
                            self._schedule_assessment()
                elif kind.endswith("output_audio_transcript.done"):
                    self._response_transcript = event.get("transcript") or self._response_transcript
                    if self._response_done and self._response_transcript and not self._response_transcript_emitted:
                        self._response_transcript_emitted = True
                        yield Transcript("agent", self._response_transcript)
                elif kind == "error":
                    message = event.get("error", {}).get("message", json.dumps(event))
                    if _is_stale_cancel_error(message):
                        self._response_active = False
                        log.info("ignoring stale realtime cancellation: %s", message)
                        continue
                    yield Failed(message)
        except websockets.ConnectionClosed as exc:
            yield Failed(f"model connection closed: {exc}")
        finally:
            self._cancel_memory_task()
            self._closed = True

    async def close(self) -> None:
        self._closed = True
        self._cancel_memory_task()
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def _send(self, payload: dict) -> None:
        assert self._ws is not None
        await self._ws.send(json.dumps(payload))

    def _begin_user_turn(self, transcript: str, *, keep_memory_task: bool = False) -> None:
        transcript = transcript.strip()
        if not transcript:
            return
        self._last_user_transcript = transcript
        self._assessment_scheduled = False
        if not keep_memory_task:
            self._cancel_memory_task()

    def _reset_for_new_speech(self) -> None:
        self._cancel_memory_task()
        self._last_user_transcript = ""
        self._assessment_scheduled = False
        self._memory_query = ""

    def _cancel_memory_task(self) -> None:
        if self._memory_task is not None and not self._memory_task.done():
            self._memory_task.cancel()
        self._memory_task = None

    def _schedule_memory_prefetch(self, transcript: str) -> None:
        transcript = transcript.strip()
        if not transcript or transcript == self._memory_query:
            return
        self._memory_query = transcript
        self._cancel_memory_task()
        self._memory_task = asyncio.create_task(self._refresh_memory_context(transcript))

    async def _refresh_memory_context(self, transcript: str) -> None:
        try:
            context = await prefetch_memory_context(transcript, self.student_id)
            next_context = context.get("weak_concepts", context)
            if isinstance(next_context, dict) and not next_context.get("found") and "error" not in next_context:
                next_context = {"found": False}
            if next_context == self._memory_context:
                return
            self._memory_context = next_context
            if self._ws is not None and not self._closed:
                await self._send({
                    "type": "session.update",
                    "session": {"instructions": agent_spec.persona(self.mode, self._memory_context)},
                })
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("realtime learner-memory prefetch failed")
        finally:
            current = asyncio.current_task()
            if self._memory_task is current:
                self._memory_task = None

    def _schedule_assessment(self) -> None:
        if self._assessment_scheduled:
            return
        user = self._last_user_transcript.strip()
        assistant = self._response_transcript.strip()
        if not user or not assistant:
            log.warning(
                "external brain not scheduled for realtime turn: user_transcript=%s assistant_transcript=%s",
                bool(user),
                bool(assistant),
            )
            return
        self._conversation.extend([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ])
        if len(self._conversation) > MAX_HISTORY_MESSAGES:
            del self._conversation[:-MAX_HISTORY_MESSAGES]
        worker = EXTERNAL_BRAIN if self.student_id == "default-student" else get_external_brain(self.student_id)
        worker.schedule(self._conversation, source="realtime")
        self._assessment_scheduled = True
        self._last_user_transcript = ""


def _transcript_from_response(event: dict) -> str:
    for item in event.get("response", {}).get("output", []):
        for content in item.get("content", []):
            transcript = content.get("transcript") or content.get("text")
            if transcript:
                return transcript
    return ""


def _is_stale_cancel_error(message: str) -> bool:
    normalized = message.casefold()
    return "cancellation failed" in normalized and "no active response found" in normalized
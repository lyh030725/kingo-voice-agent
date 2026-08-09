"""Week 3 KINGO VOICE TA: hands-free streaming, VAD, and Socratic RAG."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import time
import wave
from contextlib import asynccontextmanager
from collections import deque
from pathlib import Path

import requests
import webrtcvad
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from brain import (
    StageTimer,
    TextQuestion,
    _for_speech,
    add_course_material,
    add_trusted_domain,
    get_trusted_domains,
    list_course_materials,
    next_review_prompt,
    remove_course_material,
    remove_trusted_domain,
    require_env,
    reset_conversation,
    shutdown,
    startup,
    think,
)
from transport import (
    AgentAudio,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("listener")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start and stop shared agent resources.

    Args:
        _app: FastAPI application instance.

    Yields:
        Control while the application is serving requests.
    """
    await startup()
    try:
        yield
    finally:
        await shutdown()


app = FastAPI(title="Week 3 KINGO VOICE TA Listener", lifespan=lifespan)

COURSE_NAME = "시계열데이터처리개론"
VALID_MODES = {"explain", "socratic"}


class TrustedSite(BaseModel):
    url: str = Field(min_length=1, max_length=500)

XAI_BASE = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2

SILENCE_MS = int(os.environ.get("SILENCE_MS", "900"))
PREFIX_MS = int(os.environ.get("PREFIX_MS", "300"))
MIN_SPEECH_MS = int(os.environ.get("MIN_SPEECH_MS", "250"))
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))

# Onset debounce: declare SPEAKING only after 3 speech frames in the last 5.
ONSET_FRAMES, ONSET_WINDOW = 3, 5


# --------------------------------------------------------------------------
# Generation 2: webrtcvad + endpointing state machine.
#
# The `vad` constructor argument exists for testing: inject a stub with a
# scripted is_speech() and the state machine becomes fully deterministic.
# --------------------------------------------------------------------------

class TurnDetector:
    """Detect complete speech turns from fixed-size PCM frames."""
    def __init__(self, vad=None) -> None:
        """Create detector.

        Args:
            vad: Optional VAD-compatible detector for tests.

        Returns:
            None.
        """
        self.vad = vad or webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.reset()

    def reset(self) -> None:
        """Clear accumulated turn state.

        Returns:
            None.
        """
        self.speaking = False
        self._frames: list[bytes] = []
        self._prefix: deque[bytes] = deque(maxlen=PREFIX_MS // FRAME_MS)
        self._onset: deque[bool] = deque(maxlen=ONSET_WINDOW)
        self._quiet_ms = 0
        self._speech_ms = 0

    def feed(self, frame: bytes) -> bytes | None:
        """Consume one PCM frame.

        Args:
            frame: Exactly 20 ms of mono 16 kHz int16 PCM.

        Returns:
            Complete PCM utterance at endpoint, otherwise None.
        """
        is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

        if not self.speaking:
            # IDLE: remember recent audio (prefix padding) and debounce onset.
            self._prefix.append(frame)
            self._onset.append(is_speech)
            if sum(self._onset) >= ONSET_FRAMES:
                self.speaking = True
                self._frames = list(self._prefix)   # first syllables survive
                self._speech_ms = sum(self._onset) * FRAME_MS
                self._quiet_ms = 0
                log.info("[SPEECH START]")
            return None

        # SPEAKING: keep everything (pauses are part of the audio), count
        # trailing silence, and commit the turn at SILENCE_MS.
        self._frames.append(frame)
        if is_speech:
            self._speech_ms += FRAME_MS
            self._quiet_ms = 0
        else:
            self._quiet_ms += FRAME_MS

        if self._quiet_ms < SILENCE_MS:
            return None

        utterance = b"".join(self._frames)
        speech_ms = self._speech_ms
        duration_s = len(self._frames) * FRAME_MS / 1000
        self.reset()

        if speech_ms < MIN_SPEECH_MS:
            log.info("[DISCARDED] %d ms of speech — a cough, not a turn "
                     "(this just saved you an STT + Grok call)", speech_ms)
            return None

        log.info("[SPEECH END after %.1fs -> pipeline]", duration_s)
        return utterance


# --------------------------------------------------------------------------
# Provided plumbing — identical to the scaffold from here down.
# --------------------------------------------------------------------------

def wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw PCM in WAV.

    Args:
        pcm: Raw little-endian int16 samples.

    Returns:
        Mono 16 kHz WAV bytes.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def stt(pcm: bytes) -> str:
    """Transcribe completed audio.

    Args:
        pcm: Raw mono 16 kHz int16 utterance.

    Returns:
        Recognized transcript text.
    """
    resp = requests.post(
        f"{XAI_BASE}/stt",
        headers={"Authorization": f"Bearer {require_env('XAI_API_KEY')}"},
        files={"file": ("turn.wav", wav_bytes(pcm), "audio/wav")},
    )
    resp.raise_for_status()
    return resp.json()["text"]


def tts_stream(text: str):
    """Stream speech audio.

    Args:
        text: Full answer; URLs are removed before speech.

    Yields:
        Successive MP3 byte chunks.
    """
    resp = requests.post(
        f"{XAI_BASE}/tts",
        headers={
            "Authorization": f"Bearer {require_env('XAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "text": _for_speech(text),
            "voice_id": require_env("TTS_VOICE"),
            "language": "ko",
        },
        stream=True,
    )
    resp.raise_for_status()
    yield from resp.iter_content(chunk_size=4096)


async def respond(ws: WebSocket, utterance: bytes, mode: str = "socratic") -> bool:
    """Process and stream one turn.

    Args:
        ws: Active browser WebSocket.
        utterance: Endpointed raw PCM audio.
        mode: Teaching mode selected by learner.

    Returns:
        True when reply audio was sent, otherwise False.
    """
    t0 = time.perf_counter()
    ms = lambda: round((time.perf_counter() - t0) * 1000)  # noqa: E731
    timings: dict[str, int] = {}

    await ws.send_json({"type": "turn", "duration_ms": len(utterance) // 32})
    try:
        # The blocking calls run in a worker thread (asyncio.to_thread), NOT
        # on the event loop. This is not week-5 perfectionism — streaming
        # DEPENDS on it: a blocked loop can't flush bytes to the socket, so
        # without to_thread every "streamed" chunk would pile up in the
        # transport buffer and arrive at the browser as one lump the moment
        # the pipeline finished. (Try it: drop the to_thread wrappers and
        # watch time-to-first-audio get exactly as bad as week 2.)
        transcript = await asyncio.to_thread(stt, utterance)
        timings["stt"] = ms()
        log.info("heard: %r", transcript)
        await ws.send_json({"type": "transcript", "text": transcript})

        reply, tools_used, sources, visualizations = await think(transcript, StageTimer(), mode)
        timings["llm"] = ms() - timings["stt"]
        log.info("reply: %r (tools: %s)", reply, tools_used or "none")

        await ws.send_json({"type": "audio_start"})
        first_chunk_at = None
        chunks = tts_stream(reply)
        # next() on a requests stream blocks too — same rule, same fix.
        while (chunk := await asyncio.to_thread(next, chunks, None)) is not None:
            if first_chunk_at is None:
                first_chunk_at = ms()
                timings["tts_first"] = first_chunk_at - timings["stt"] - timings["llm"]
            await ws.send_json({
                "type": "audio_chunk",
                "data": base64.b64encode(chunk).decode(),
            })
        timings["tts_total"] = ms() - timings["stt"] - timings["llm"]
        timings["total"] = ms()
        log.info("timings: %s", timings)

        await ws.send_json({
            "type": "audio_end",
            "reply": reply,
            "tools": tools_used,
            "sources": sources,
            "visualizations": visualizations,
            "timings": timings,
        })
        return True
    except Exception as exc:
        log.exception("pipeline failed")
        await ws.send_json({"type": "error", "message": str(exc)})
        return False


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    mode = ws.query_params.get("mode", "socratic")
    if mode not in VALID_MODES:
        mode = "socratic"
    transport = make_transport(mode)
    reader = asyncio.create_task(pump_provider_events(ws, transport))
    try:
        await transport.start()
        await ws.send_json({"type": "ready", "provider": transport.name})
        await pump_caller_audio(ws, transport)
    except WebSocketDisconnect:
        log.info("stream closed")
    except Exception as exc:
        log.exception("realtime voice failed")
        await try_send(ws, {"type": "error", "message": str(exc)})
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await transport.close()


def make_transport(mode: str) -> Transport:
    """Create the Week 4 Grok realtime transport."""
    from grok_live import GrokTransport

    return GrokTransport(mode)


async def pump_caller_audio(ws: WebSocket, transport: Transport) -> None:
    """Relay validated 20 ms PCM frames while the mic stays open."""
    while True:
        msg = await ws.receive_json()
        if not isinstance(msg, dict) or msg.get("type") != "audio":
            continue
        try:
            frame = base64.b64decode(msg.get("data", ""), validate=True)
        except (binascii.Error, TypeError, ValueError):
            continue
        if len(frame) == FRAME_BYTES:
            await transport.send_audio(frame)


async def pump_provider_events(ws: WebSocket, transport: Transport) -> None:
    """Translate provider-neutral realtime events for the existing browser."""
    speaking = False
    turn_ended_at: float | None = None
    try:
        async for event in transport.events():
            match event:
                case SessionReady():
                    log.info("realtime session configured")
                case UserStartedSpeaking():
                    await ws.send_json({"type": "state", "value": "hearing"})
                    speaking = False
                    await ws.send_json({"type": "flush"})
                case UserStoppedSpeaking():
                    turn_ended_at = time.perf_counter()
                    await ws.send_json({"type": "state", "value": "thinking"})
                case AgentAudio(pcm=pcm, rate=rate):
                    if not speaking:
                        speaking = True
                        await ws.send_json({"type": "state", "value": "speaking"})
                        if turn_ended_at is not None:
                            await ws.send_json({
                                "type": "latency",
                                "ms": round((time.perf_counter() - turn_ended_at) * 1000),
                            })
                            turn_ended_at = None
                    await ws.send_json({
                        "type": "audio",
                        "data": base64.b64encode(pcm).decode(),
                        "rate": rate,
                    })
                case AgentTextDelta(text=text) if text:
                    await ws.send_json({"type": "token", "text": text})
                case AgentTurnDone():
                    speaking = False
                    await ws.send_json({"type": "turn_done"})
                case Transcript(who=who, text=text) if text:
                    await ws.send_json({"type": "transcript", "who": who, "text": text})
                case ToolCalled(name=name, result=result):
                    await ws.send_json({"type": "tool", "name": name})
                    if name == "show_visualization" and isinstance(result, dict):
                        await ws.send_json({"type": "visualization", "visualization": result})
                case Failed(message=message):
                    await ws.send_json({"type": "error", "message": message})
                    return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("provider event pump failed")
        await try_send(ws, {"type": "error", "message": str(exc)})


async def try_send(ws: WebSocket, payload: dict) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        pass


@app.post("/answer-text")
async def answer_text(question: TextQuestion) -> dict:
    """Run brain without audio.

    Args:
        question: Validated text question.

    Returns:
        Reply, tools, trusted sources, transcript, and timings.
    """
    transcript = question.text.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="text must not be blank")
    timer = StageTimer()
    try:
        with timer.stage("total"):
            reply, tools_used, sources, visualizations = await think(transcript, timer, question.mode)
    except Exception as exc:
        log.exception("text answer failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "transcript": transcript,
        "reply": reply,
        "tools": tools_used,
        "sources": sources,
        "visualizations": visualizations,
        "timings": timer.timings_ms,
    }


@app.post("/answer-text/stream")
async def answer_text_stream(question: TextQuestion) -> StreamingResponse:
    """Stream model text deltas followed by one result event."""
    transcript = question.text.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="text must not be blank")

    async def events():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        timer = StageTimer()

        async def on_token(token: str) -> None:
            await queue.put({"type": "token", "text": token})

        async def generate() -> None:
            try:
                with timer.stage("total"):
                    reply, tools_used, sources, visualizations = await think(
                        transcript, timer, question.mode, on_token=on_token
                    )
                await queue.put({
                    "type": "done",
                    "transcript": transcript,
                    "reply": reply,
                    "tools": tools_used,
                    "sources": sources,
                    "visualizations": visualizations,
                    "timings": timer.timings_ms,
                })
            except Exception as exc:
                log.exception("streaming text answer failed")
                await queue.put({"type": "error", "message": str(exc)})

        task = asyncio.create_task(generate())
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] in {"done", "error"}:
                    break
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/reset")
async def reset() -> dict:
    """Reset current conversation.

    Returns:
        Success marker; persistent weak concepts remain.
    """
    reset_conversation()
    return {"ok": True}


@app.get("/review")
async def review() -> dict:
    """Fetch next due review.

    Returns:
        Due flag and optional weak-concept question.
    """
    return await next_review_prompt()



@app.get("/api/materials")
async def materials() -> dict:
    """List course PDFs available to the agent.

    Returns:
        Course name and uploaded material records.
    """
    return {"course": COURSE_NAME, "materials": list_course_materials()}


@app.post("/api/materials")
async def upload_material(request: Request) -> dict:
    """Upload one raw PDF request body.

    Args:
        request: Request with filename query parameter and PDF body.

    Returns:
        Saved material record.
    """
    filename = request.query_params.get("filename", "")
    content = await request.body()
    if not content:
        raise HTTPException(status_code=422, detail="PDF file is required")
    try:
        material = add_course_material(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "material": material}


@app.delete("/api/materials")
async def delete_material(request: Request) -> dict:
    """Delete one professor-uploaded PDF."""
    filename = request.query_params.get("filename", "")
    try:
        remove_course_material(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="course material not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.get("/api/trusted-sites")
async def trusted_sites() -> dict:
    """List professor-managed web-search domains.

    Returns:
        Current trusted domain allowlist.
    """
    return {"sites": get_trusted_domains()}


@app.post("/api/trusted-sites")
async def create_trusted_site(site: TrustedSite) -> dict:
    """Add one trusted web-search domain.

    Args:
        site: URL or hostname supplied by professor.

    Returns:
        Updated trusted domain allowlist.
    """
    try:
        return {"sites": add_trusted_domain(site.url)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/trusted-sites")
async def delete_trusted_site(site: TrustedSite) -> dict:
    """Delete one trusted web-search domain.

    Args:
        site: URL or hostname supplied by professor.

    Returns:
        Updated trusted domain allowlist.
    """
    try:
        return {"sites": remove_trusted_domain(site.url)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


BASE_DIR = Path(__file__).resolve().parent
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")

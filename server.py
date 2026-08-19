"""KINGO Voice Agent FastAPI server."""

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
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import webrtcvad
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from brain import (
    StageTimer,
    TextQuestion,
    add_course_material,
    add_trusted_domain,
    get_course_material_path,
    get_trusted_domains,
    list_course_materials,
    list_weak_concepts,
    next_review_prompt,
    remove_course_material,
    remove_trusted_domain,
    reset_conversation,
    shutdown,
    startup,
    think,
)
from session_state import clean_id
from transport import (
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("listener")

STUDENT_COOKIE = "kingo_student_id"
SESSION_COOKIE = "kingo_session_id"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365
ACTIVE_REALTIME: dict[tuple[str, str], tuple[WebSocket, Transport]] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await startup()
    try:
        yield
    finally:
        await shutdown()


app = FastAPI(title="KINGO VOICE TA", lifespan=lifespan)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _identity(request: Request | None) -> tuple[str, str]:
    if request is None:
        return "default-student", "default-session"
    student_id = getattr(request.state, "student_id", None)
    session_id = getattr(request.state, "session_id", None)
    return (
        clean_id(student_id or request.cookies.get(STUDENT_COOKIE), "default-student"),
        clean_id(session_id or request.cookies.get(SESSION_COOKIE), "default-session"),
    )


@app.middleware("http")
async def ensure_browser_identity(request: Request, call_next):
    """Assign an anonymous learner identity and a replaceable chat-session id."""
    raw_student = request.cookies.get(STUDENT_COOKIE)
    raw_session = request.cookies.get(SESSION_COOKIE)
    student_id = clean_id(raw_student, "") or _new_id("student")
    session_id = clean_id(raw_session, "") or _new_id("session")
    request.state.student_id = student_id
    request.state.session_id = session_id

    response = await call_next(request)
    if raw_student != student_id:
        response.set_cookie(
            STUDENT_COOKIE,
            student_id,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
    if raw_session != session_id:
        response.set_cookie(
            SESSION_COOKIE,
            session_id,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
    return response


COURSE_NAME = "시계열데이터처리개론"
VALID_MODES = {"explain", "socratic"}


class TrustedSite(BaseModel):
    url: str = Field(min_length=1, max_length=500)


SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2
SILENCE_MS = int(os.environ.get("SILENCE_MS", "900"))
PREFIX_MS = int(os.environ.get("PREFIX_MS", "300"))
MIN_SPEECH_MS = int(os.environ.get("MIN_SPEECH_MS", "250"))
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))
ONSET_FRAMES, ONSET_WINDOW = 3, 5


class TurnDetector:
    """Legacy deterministic endpoint detector retained for course tests."""

    def __init__(self, vad=None) -> None:
        self.vad = vad or webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.reset()

    def reset(self) -> None:
        self.speaking = False
        self._frames: list[bytes] = []
        self._prefix: deque[bytes] = deque(maxlen=PREFIX_MS // FRAME_MS)
        self._onset: deque[bool] = deque(maxlen=ONSET_WINDOW)
        self._quiet_ms = 0
        self._speech_ms = 0

    def feed(self, frame: bytes) -> bytes | None:
        is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
        if not self.speaking:
            self._prefix.append(frame)
            self._onset.append(is_speech)
            if sum(self._onset) >= ONSET_FRAMES:
                self.speaking = True
                self._frames = list(self._prefix)
                self._speech_ms = sum(self._onset) * FRAME_MS
                self._quiet_ms = 0
            return None

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
        self.reset()
        if speech_ms < MIN_SPEECH_MS:
            return None
        return utterance


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(pcm)
    return buf.getvalue()


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    mode = ws.query_params.get("mode", "socratic")
    if mode not in VALID_MODES:
        mode = "socratic"
    student_id = clean_id(ws.cookies.get(STUDENT_COOKIE), "default-student")
    session_id = clean_id(ws.cookies.get(SESSION_COOKIE), "default-session")
    transport = make_transport(mode, student_id, session_id)
    key = (student_id, session_id)
    ACTIVE_REALTIME[key] = (ws, transport)
    reader = asyncio.create_task(pump_provider_events(ws, transport))
    try:
        await transport.start()
        await ws.send_json({"type": "ready", "provider": transport.name})
        await pump_caller_input(ws, transport)
    except WebSocketDisconnect:
        log.info("stream closed student=%s session=%s", student_id, session_id)
    except Exception as exc:
        log.exception("realtime voice failed")
        await try_send(ws, {"type": "error", "message": str(exc)})
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        current = ACTIVE_REALTIME.get(key)
        if current is not None and current[1] is transport:
            ACTIVE_REALTIME.pop(key, None)
        await transport.close()


def make_transport(
    mode: str,
    student_id: str = "default-student",
    session_id: str = "default-session",
) -> Transport:
    from grok_live import GrokTransport

    return GrokTransport(mode, student_id=student_id, session_id=session_id)


async def pump_caller_input(ws: WebSocket, transport: Transport) -> None:
    while True:
        msg = await ws.receive_json()
        if not isinstance(msg, dict):
            continue
        if msg.get("type") == "text":
            text = msg.get("text")
            if isinstance(text, str) and 1 <= len(text.strip()) <= 4000:
                await ws.send_json({"type": "state", "value": "thinking"})
                await transport.send_text(text.strip())
            continue
        if msg.get("type") != "audio":
            continue
        try:
            frame = base64.b64decode(msg.get("data", ""), validate=True)
        except (binascii.Error, TypeError, ValueError):
            continue
        if len(frame) == FRAME_BYTES:
            await transport.send_audio(frame)


async def pump_provider_events(ws: WebSocket, transport: Transport) -> None:
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
                            await ws.send_json(
                                {
                                    "type": "latency",
                                    "ms": round(
                                        (time.perf_counter() - turn_ended_at) * 1000
                                    ),
                                }
                            )
                            turn_ended_at = None
                    await ws.send_json(
                        {
                            "type": "audio",
                            "data": base64.b64encode(pcm).decode(),
                            "rate": rate,
                        }
                    )
                case AgentTextDelta(text=text) if text:
                    await ws.send_json({"type": "token", "text": text})
                case AgentTextBoundary():
                    await ws.send_json({"type": "text_boundary"})
                case AgentTurnDone():
                    speaking = False
                    await ws.send_json({"type": "turn_done"})
                case Transcript(who=who, text=text) if text:
                    await ws.send_json(
                        {"type": "transcript", "who": who, "text": text}
                    )
                case ToolCalled(name=name, result=result):
                    await ws.send_json({"type": "tool", "name": name})
                    if name == "show_visualization" and isinstance(result, dict):
                        if "error" in result:
                            await ws.send_json(
                                {
                                    "type": "visualization_error",
                                    "message": (
                                        "시각자료를 표시하지 못했어요. 다시 요청해 주세요."
                                    ),
                                }
                            )
                        else:
                            await ws.send_json(
                                {
                                    "type": "visualization",
                                    "visualization": result,
                                }
                            )
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
async def answer_text(question: TextQuestion, request: Request = None) -> dict:
    transcript = question.text.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="text must not be blank")
    student_id, session_id = _identity(request)
    timer = StageTimer()
    try:
        with timer.stage("total"):
            reply, tools_used, sources, visualizations = await think(
                transcript,
                timer,
                question.mode,
                student_id=student_id,
                session_id=session_id,
            )
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
async def answer_text_stream(
    question: TextQuestion, request: Request = None
) -> StreamingResponse:
    transcript = question.text.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="text must not be blank")
    student_id, session_id = _identity(request)

    async def events():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        timer = StageTimer()

        async def on_token(token: str) -> None:
            await queue.put({"type": "token", "text": token})

        async def generate() -> None:
            try:
                with timer.stage("total"):
                    reply, tools_used, sources, visualizations = await think(
                        transcript,
                        timer,
                        question.mode,
                        on_token=on_token,
                        student_id=student_id,
                        session_id=session_id,
                    )
                await queue.put(
                    {
                        "type": "done",
                        "transcript": transcript,
                        "reply": reply,
                        "tools": tools_used,
                        "sources": sources,
                        "visualizations": visualizations,
                        "timings": timer.timings_ms,
                    }
                )
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
async def reset(request: Request, response: Response) -> dict:
    """Start a new chat while preserving the learner's persistent memory."""
    student_id, session_id = _identity(request)
    reset_conversation(student_id, session_id)
    active = ACTIVE_REALTIME.pop((student_id, session_id), None)
    if active is not None:
        browser, transport = active
        await transport.close()
        try:
            await browser.close(code=1000)
        except Exception:
            pass

    next_session = _new_id("session")
    response.set_cookie(
        SESSION_COOKIE,
        next_session,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    request.state.session_id = next_session
    return {"ok": True, "session_reset": True}


@app.get("/review")
async def review(request: Request = None) -> dict:
    student_id, session_id = _identity(request)
    return await next_review_prompt(student_id, session_id)


@app.get("/api/weak-concepts")
async def weak_concepts(request: Request = None) -> dict:
    student_id, _ = _identity(request)
    return {"concepts": await list_weak_concepts(student_id)}


@app.get("/api/materials")
async def materials() -> dict:
    return {"course": COURSE_NAME, "materials": list_course_materials()}


@app.get("/api/materials/{filename}", response_class=FileResponse)
async def course_material_file(filename: str) -> FileResponse:
    try:
        path = get_course_material_path(filename)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="course material not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
        content_disposition_type="inline",
    )


@app.post("/api/materials")
async def upload_material(request: Request) -> dict:
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
    filename = request.query_params.get("filename", "")
    try:
        remove_course_material(filename)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="course material not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.get("/api/trusted-sites")
async def trusted_sites() -> dict:
    return {"sites": get_trusted_domains()}


@app.post("/api/trusted-sites")
async def create_trusted_site(site: TrustedSite) -> dict:
    try:
        return {"sites": add_trusted_domain(site.url)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/trusted-sites")
async def delete_trusted_site(site: TrustedSite) -> dict:
    try:
        return {"sites": remove_trusted_domain(site.url)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


BASE_DIR = Path(__file__).resolve().parent
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")

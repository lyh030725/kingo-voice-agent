"""Provider-neutral contract for live speech-to-speech sessions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

CALLER_RATE = 16000
AGENT_RATE = 24000


@dataclass(frozen=True)
class SessionReady:
    pass


@dataclass(frozen=True)
class UserStartedSpeaking:
    pass


@dataclass(frozen=True)
class UserStoppedSpeaking:
    pass


@dataclass(frozen=True)
class AgentAudio:
    pcm: bytes
    rate: int = AGENT_RATE


@dataclass(frozen=True)
class AgentTextDelta:
    text: str


@dataclass(frozen=True)
class AgentTextBoundary:
    pass


@dataclass(frozen=True)
class AgentTurnDone:
    pass


@dataclass(frozen=True)
class Transcript:
    who: str
    text: str


@dataclass(frozen=True)
class ToolCalled:
    name: str
    args: dict = field(default_factory=dict)
    result: object = None


@dataclass(frozen=True)
class Failed:
    message: str


Event = SessionReady | UserStartedSpeaking | UserStoppedSpeaking | AgentAudio | AgentTextDelta | AgentTextBoundary | AgentTurnDone | Transcript | ToolCalled | Failed


class Transport(ABC):
    name = "transport"

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None:
        pass

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Submit a typed user turn to the live voice session."""
        pass

    @abstractmethod
    def events(self) -> AsyncIterator[Event]:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

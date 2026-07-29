from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["VOICE_AI_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


class FakeVad:
    def __init__(self, decisions: list[bool]) -> None:
        self.decisions = iter(decisions)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self.last_call = (len(frame), sample_rate)
        return next(self.decisions)


FRAME = bytes(server.FRAME_BYTES)


class TurnDetectorTests(unittest.TestCase):
    def test_debounce_prefix_and_endpoint(self) -> None:
        decisions = [False, False, True, True, True] + [True] * 10 + [False] * 45
        detector = server.TurnDetector(FakeVad(decisions))
        utterance = None
        for _ in decisions:
            utterance = detector.feed(FRAME) or utterance

        self.assertIsNotNone(utterance)
        self.assertEqual(len(utterance), len(decisions) * server.FRAME_BYTES)
        self.assertFalse(detector.speaking)

    def test_silence_must_be_consecutive(self) -> None:
        decisions = [True, True, True, False, False, True, False, False, False]
        with (
            patch.object(server, "PREFIX_MS", 40),
            patch.object(server, "SILENCE_MS", 60),
            patch.object(server, "MIN_SPEECH_MS", 60),
        ):
            detector = server.TurnDetector(FakeVad(decisions))
            results = [detector.feed(FRAME) for _ in decisions]

        self.assertTrue(all(result is None for result in results[:-1]))
        self.assertIsNotNone(results[-1])

    def test_short_noise_is_discarded(self) -> None:
        decisions = [True, True, True, False, False]
        with (
            patch.object(server, "PREFIX_MS", 40),
            patch.object(server, "SILENCE_MS", 40),
            patch.object(server, "MIN_SPEECH_MS", 100),
        ):
            detector = server.TurnDetector(FakeVad(decisions))
            results = [detector.feed(FRAME) for _ in decisions]

        self.assertTrue(all(result is None for result in results))
        self.assertFalse(detector.speaking)

    def test_wav_contract(self) -> None:
        wav = server.wav_bytes(FRAME)
        self.assertEqual(wav[:4], b"RIFF")
        self.assertIn(b"WAVE", wav[:16])


if __name__ == "__main__":
    unittest.main()

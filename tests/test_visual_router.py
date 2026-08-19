from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ["VOICE_AI_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import visual_router


class VisualRouterTests(unittest.TestCase):
    def test_decision_request_forces_one_specific_tool(self) -> None:
        history = [
            {"role": "assistant", "content": "Scaled Dot-Product Attention 전체 수식으로 넘어가 볼까요?"}
        ]
        request = visual_router._decision_request("네, 설명해 줘요.", history)

        self.assertEqual(
            request["tool_choice"],
            {"type": "function", "function": {"name": "decide_visual"}},
        )
        self.assertFalse(request["parallel_tool_calls"])
        self.assertEqual(len(request["tools"]), 1)
        self.assertEqual(request["tools"][0]["function"]["name"], "decide_visual")

        context = json.loads(request["messages"][1]["content"])
        self.assertEqual(context["current_user"], "네, 설명해 줘요.")
        self.assertIn("Scaled Dot-Product Attention", context["recent_conversation"][0]["content"])

    def test_formula_decision_returns_render_ready_payload(self) -> None:
        payload = {
            "needed": True,
            "kind": "formula",
            "title": "스케일드 닷 프로덕트 어텐션",
            "caption": "점곱을 차원 크기로 스케일링해요.",
            "latex": r"\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^T/\sqrt{d_k})V",
            "labels": [],
            "file": "",
            "page": 0,
        }
        message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(arguments=json.dumps(payload, ensure_ascii=False))
                )
            ]
        )

        decision = visual_router._parse_decision(message)

        self.assertTrue(decision.needed)
        self.assertEqual(decision.kind, "formula")
        self.assertEqual(decision.args["kind"], "formula")
        self.assertIn("softmax", decision.args["latex"])
        self.assertNotIn("labels", decision.args)

    def test_none_decision_does_not_create_visual(self) -> None:
        payload = {
            "needed": False,
            "kind": "none",
            "title": "",
            "caption": "",
            "latex": "",
            "labels": [],
            "file": "",
            "page": 0,
        }
        message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(arguments=json.dumps(payload))
                )
            ]
        )

        self.assertFalse(visual_router._parse_decision(message).needed)


if __name__ == "__main__":
    unittest.main()

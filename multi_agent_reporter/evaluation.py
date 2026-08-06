from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .verification import ClaimCheck, verification_metrics


@dataclass(slots=True)
class ReportEvaluation:
    report: str
    checks: list[ClaimCheck]
    metrics: dict[str, float]
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.33))


def evaluate_report(report: str, sources) -> ReportEvaluation:
    checks = __import__("multi_agent_reporter.verification", fromlist=["verify_report"]).verify_report(report, sources)
    return ReportEvaluation(report, checks, verification_metrics(checks), estimate_tokens(report))


def _parse_score(text: str) -> float | None:
    try:
        start, end = text.find("{"), text.rfind("}")
        payload = json.loads(text[start : end + 1])
        values = [float(payload[key]) for key in ("report_a", "report_b")]
        return values[0] - values[1]
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def judge_pair(model, report_a: str, report_b: str, task: str, seed: int = 42) -> dict[str, float | str]:
    """Use anonymous labels, swap answer order, and average only stable judgments."""
    prompt = (
        "You are a blinded evaluator. Score each report independently from 0 to 10 for factuality, "
        "clarity and completeness. Ignore length and presentation order. Return strict JSON: "
        '{"report_a": 0.0, "report_b": 0.0}.\nTASK: ' + task
    )
    rng = random.Random(seed)
    first = model.invoke(prompt + "\nREPORT A:\n" + report_a + "\nREPORT B:\n" + report_b)
    second = model.invoke(prompt + "\nREPORT A:\n" + report_b + "\nREPORT B:\n" + report_a)
    first_text = getattr(first, "content", str(first))
    second_text = getattr(second, "content", str(second))
    delta_one, delta_two = _parse_score(first_text), _parse_score(second_text)
    if delta_one is None or delta_two is None:
        return {"status": "judge_parse_failed", "position_consistency": 0.0}
    # In the swapped call, report A/B are reversed, so the expected delta changes sign.
    consistency = 1.0 if abs(delta_one + delta_two) < 0.5 else 0.0
    return {
        "report_a_minus_b": (delta_one - delta_two) / 2,
        "position_consistency": consistency,
        "status": "stable" if consistency else "position_sensitive",
    }


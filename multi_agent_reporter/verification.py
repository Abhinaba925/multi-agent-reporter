from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieval import RetrievedSource


@dataclass(slots=True)
class ClaimCheck:
    claim: str
    citation_ids: list[str]
    status: str
    support_score: float
    reason: str


def extract_claims(report: str) -> list[tuple[str, list[str]]]:
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+", report):
        sentence = sentence.strip()
        if len(sentence.split()) < 5 or sentence.startswith("#"):
            continue
        citations = sorted(set(re.findall(r"\[(S\d+)\]", sentence)))
        claims.append((re.sub(r"\s*\[S\d+\]", "", sentence), citations))
    return claims


def verify_report(report: str, sources: list[RetrievedSource]) -> list[ClaimCheck]:
    source_map = {source.source_id: source for source in sources}
    checks = []
    for claim, citation_ids in extract_claims(report):
        if not citation_ids:
            checks.append(ClaimCheck(claim, [], "missing_citation", 0.0, "No source citation was attached."))
            continue
        known = [source_map[item] for item in citation_ids if item in source_map]
        if not known:
            checks.append(ClaimCheck(claim, citation_ids, "invalid_citation", 0.0, "Citation ID is not in retrieved evidence."))
            continue
        claim_words = set(re.findall(r"[a-z0-9]+", claim.lower()))
        best = 0.0
        for source in known:
            evidence_words = set(re.findall(r"[a-z0-9]+", (source.content or source.snippet).lower()))
            best = max(best, len(claim_words & evidence_words) / max(1, len(claim_words)))
        status = "supported" if best >= 0.28 else "partially_supported" if best >= 0.12 else "unsupported"
        checks.append(ClaimCheck(claim, citation_ids, status, best, "Lexical claim-to-evidence check."))
    return checks


def verification_metrics(checks: list[ClaimCheck]) -> dict[str, float]:
    if not checks:
        return {"citation_completeness": 0.0, "claim_support_precision": 0.0, "unsupported_claim_rate": 1.0}
    cited = sum(bool(item.citation_ids) for item in checks)
    supported = sum(item.status == "supported" for item in checks)
    unsupported = sum(item.status in {"unsupported", "invalid_citation", "missing_citation"} for item in checks)
    return {
        "citation_completeness": cited / len(checks),
        "claim_support_precision": supported / len(checks),
        "unsupported_claim_rate": unsupported / len(checks),
    }


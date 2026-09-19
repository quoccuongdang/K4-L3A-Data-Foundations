"""Deterministic evaluation metrics for the FPTU retrieval benchmark."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


def normalize_text(value: str) -> str:
    """Normalize Vietnamese text for robust phrase and keyword matching."""
    value = value.replace("đ", "d").replace("Đ", "D")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower()
    # Remove Markdown markers and punctuation so ``**Orientation**`` matches
    # the human-authored gold phrase ``Orientation``.
    value = re.sub(r"[^a-z0-9%]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _contains(text: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(text)


def matched_evidence_ids(text: str, benchmark_case: dict[str, Any]) -> set[str]:
    """Return evidence-unit IDs supported by a chunk of text."""
    matched: set[str] = set()
    for unit in benchmark_case["evidence"]:
        if any(_contains(text, phrase) for phrase in unit["phrases"]):
            matched.add(unit["id"])
    return matched


def relevance_grade(text: str, benchmark_case: dict[str, Any]) -> int:
    """Graded relevance equals the number of distinct gold evidence units found."""
    return len(matched_evidence_ids(text, benchmark_case))


def evidence_recall_at(
    results: list[dict[str, Any]],
    benchmark_case: dict[str, Any],
    k: int,
) -> float:
    """Fraction of required gold evidence units covered within top-k chunks."""
    expected = {unit["id"] for unit in benchmark_case["evidence"]}
    if not expected:
        return 0.0
    retrieved: set[str] = set()
    for result in results[:k]:
        retrieved.update(matched_evidence_ids(result["content"], benchmark_case))
    return len(retrieved) / len(expected)


def reciprocal_rank(results: list[dict[str, Any]], benchmark_case: dict[str, Any]) -> float:
    """Reciprocal rank of the first chunk containing at least one evidence unit."""
    for rank, result in enumerate(results, 1):
        if relevance_grade(result["content"], benchmark_case) > 0:
            return 1.0 / rank
    return 0.0


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))


def ndcg_at(
    results: list[dict[str, Any]],
    corpus_chunks: list[Any],
    benchmark_case: dict[str, Any],
    k: int = 5,
) -> float:
    """Compute graded nDCG@k against every labeled-relevant corpus chunk."""
    actual = [relevance_grade(result["content"], benchmark_case) for result in results[:k]]
    ideal = sorted(
        (relevance_grade(chunk.content, benchmark_case) for chunk in corpus_chunks),
        reverse=True,
    )[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(actual) / ideal_dcg if ideal_dcg else 0.0


def full_evidence_at(
    results: list[dict[str, Any]],
    benchmark_case: dict[str, Any],
    k: int = 5,
) -> float:
    """Return 1 when top-k collectively covers every required evidence unit."""
    return float(evidence_recall_at(results, benchmark_case, k) == 1.0)


def agent_accuracy(answer: str, benchmark_case: dict[str, Any]) -> float:
    """Gold-keyword coverage used for the agreed Faithfulness/Agent Accuracy metric."""
    criteria = benchmark_case.get("answer_criteria", [])
    if not criteria:
        return 0.0
    normalized_answer = normalize_text(answer)
    passed = 0
    for criterion in criteria:
        terms = [normalize_text(term) for term in criterion.get("all_terms", [])]
        if terms and all(term in normalized_answer for term in terms):
            passed += 1
    return passed / len(criteria)


def audience_match_rate(
    results: list[dict[str, Any]],
    expected_audience: str,
    k: int = 5,
) -> float:
    """Share of top-k chunks meant for the expected audience (or for everyone)."""
    selected = results[:k]
    if not selected:
        return 0.0
    accepted = {expected_audience, "all"}
    matches = sum(
        1 for result in selected if result.get("metadata", {}).get("audience") in accepted
    )
    return matches / len(selected)


def evaluate_query(
    results: list[dict[str, Any]],
    corpus_chunks: list[Any],
    benchmark_case: dict[str, Any],
    answer: str,
) -> dict[str, float]:
    """Compute the seven metrics agreed by the team for one query."""
    return {
        "recall_at_1": evidence_recall_at(results, benchmark_case, 1),
        "recall_at_5": evidence_recall_at(results, benchmark_case, 5),
        "mrr": reciprocal_rank(results, benchmark_case),
        "ndcg_at_5": ndcg_at(results, corpus_chunks, benchmark_case, 5),
        "full_evidence_at_5": full_evidence_at(results, benchmark_case, 5),
        "faithfulness_agent_accuracy": agent_accuracy(answer, benchmark_case),
        "audience_match_rate": audience_match_rate(
            results,
            benchmark_case["expected_audience"],
            5,
        ),
    }


def macro_average(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    return {
        metric: sum(result[metric] for result in per_query) / len(per_query)
        for metric in per_query[0]
    }

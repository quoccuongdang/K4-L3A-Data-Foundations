import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chat import load_docs
from src import RecursiveChunker
from src.benchmark import (
    agent_accuracy,
    audience_match_rate,
    evidence_recall_at,
    full_evidence_at,
    macro_average,
    matched_evidence_ids,
    ndcg_at,
    normalize_text,
    reciprocal_rank,
)

ROOT = Path(__file__).parent.parent
CASES = json.loads((ROOT / "benchmark" / "gold_queries.json").read_text(encoding="utf-8"))


def case_with_two_units():
    return {
        "expected_audience": "student",
        "evidence": [
            {"id": "a", "phrases": ["bằng chứng một"]},
            {"id": "b", "phrases": ["bằng chứng hai"]},
        ],
        "answer_criteria": [
            {"id": "c1", "all_terms": ["đáp án", "một"]},
            {"id": "c2", "all_terms": ["đáp án", "hai"]},
        ],
    }


def result(content, audience="student"):
    return {"content": content, "metadata": {"audience": audience}}


def test_normalize_text_handles_vietnamese_diacritics():
    assert normalize_text("Điều kiện TỐT NGHIỆP") == "dieu kien tot nghiep"


def test_recall_and_full_evidence_use_evidence_units():
    case = case_with_two_units()
    results = [result("Bằng chứng một"), result("Bằng chứng hai")]
    assert evidence_recall_at(results, case, 1) == 0.5
    assert evidence_recall_at(results, case, 5) == 1.0
    assert full_evidence_at(results, case, 5) == 1.0


def test_mrr_uses_first_evidence_bearing_chunk():
    case = case_with_two_units()
    results = [result("không liên quan"), result("bằng chứng hai")]
    assert reciprocal_rank(results, case) == 0.5


def test_ndcg_is_one_for_ideal_order():
    case = case_with_two_units()
    results = [result("bằng chứng một và bằng chứng hai"), result("bằng chứng một")]
    corpus = [
        SimpleNamespace(content="bằng chứng một và bằng chứng hai"),
        SimpleNamespace(content="bằng chứng một"),
        SimpleNamespace(content="không liên quan"),
    ]
    assert ndcg_at(results, corpus, case, 5) == pytest.approx(1.0)


def test_agent_accuracy_is_gold_criterion_coverage():
    case = case_with_two_units()
    answer = "Đáp án một đã có, nhưng chưa có phần còn lại."
    assert agent_accuracy(answer, case) == 0.5


def test_audience_match_accepts_expected_and_all():
    results = [result("a", "student"), result("b", "all"), result("c", "faculty")]
    assert audience_match_rate(results, "student", 5) == pytest.approx(2 / 3)


def test_macro_average():
    rows = [{"m": 1.0}, {"m": 0.5}, {"m": 0.0}]
    assert macro_average(rows) == {"m": 0.5}


def test_every_gold_evidence_unit_exists_in_chunked_corpus():
    chunker = RecursiveChunker(chunk_size=600)
    chunks = [chunk for doc in load_docs() for chunk in chunker.chunk(doc.content)]
    for benchmark_case in CASES:
        found = set()
        for chunk in chunks:
            found.update(matched_evidence_ids(chunk, benchmark_case))
        expected = {unit["id"] for unit in benchmark_case["evidence"]}
        assert found == expected, f"{benchmark_case['id']} has ungrounded gold evidence"

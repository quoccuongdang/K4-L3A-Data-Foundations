"""Run the five-query FPTU benchmark with the team's seven agreed metrics.

Examples:
    python bench.py
    python bench.py --audience-filter
    python bench.py --provider mock --retrieval-only
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from chat import call_gemini_llm, get_cached_embedder, load_docs
from src import Document, EmbeddingStore, RecursiveChunker
from src.benchmark import evaluate_query, macro_average, matched_evidence_ids
from src.embeddings import GeminiEmbedder, LocalEmbedder, OpenAIEmbedder, _mock_embed

ROOT = Path(__file__).parent
GOLD_FILE = ROOT / "benchmark" / "gold_queries.json"
DEFAULT_TEXT_OUTPUT = ROOT / "ket_qua_benchmark.txt"
DEFAULT_JSON_OUTPUT = ROOT / "benchmark" / "results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("gemini", "local", "openai", "mock"),
        default=None,
        help="Embedding backend; defaults to EMBEDDING_PROVIDER from .env.",
    )
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument(
        "--audience-filter",
        action="store_true",
        help="Pre-filter every query with its expected audience before retrieval.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip Gemini answer generation; Agent Accuracy is reported as N/A.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def load_benchmark_cases() -> list[dict[str, Any]]:
    return json.loads(GOLD_FILE.read_text(encoding="utf-8"))


def select_embedder(provider: str) -> tuple[Callable[[str], list[float]], str]:
    if provider == "gemini":
        raw = GeminiEmbedder()
    elif provider == "local":
        raw = LocalEmbedder()
    elif provider == "openai":
        raw = OpenAIEmbedder()
    else:
        raw = _mock_embed
    embedder = get_cached_embedder(raw)
    return embedder, getattr(embedder, "_backend_name", provider)


def chunk_corpus(chunk_size: int) -> list[Document]:
    chunker = RecursiveChunker(chunk_size=chunk_size)
    chunks: list[Document] = []
    for doc in load_docs():
        for index, content in enumerate(chunker.chunk(doc.content)):
            chunks.append(
                Document(
                    id=f"{doc.id}_c{index}",
                    content=content,
                    metadata={**doc.metadata, "doc_id": doc.id, "chunk_idx": str(index)},
                )
            )
    return chunks


def build_benchmark_store(
    chunks: list[Document],
    embedder: Callable[[str], list[float]],
) -> EmbeddingStore:
    store = EmbeddingStore(collection_name="fptu_benchmark", embedding_fn=embedder)
    total = len(chunks)
    for index, chunk in enumerate(chunks, 1):
        store.add_documents([chunk])
        if index % 25 == 0 or index == total:
            print(f"  Đã lập chỉ mục {index}/{total} chunks")
    if hasattr(embedder, "save_cache"):
        embedder.save_cache()
    return store


def generate_agent_answer(question: str, results: list[dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"[{rank}] Nguồn: {result['metadata'].get('source', '?')}\n{result['content']}"
        for rank, result in enumerate(results, 1)
    )
    prompt = f"""Bạn là trợ lý sinh viên FPTU. Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.
Nếu ngữ cảnh thiếu thông tin, hãy nói rõ phần còn thiếu. Trả lời ngắn gọn, chính xác bằng tiếng Việt.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:"""
    return call_gemini_llm(prompt)


def serialize_result(
    rank: int,
    result: dict[str, Any],
    benchmark_case: dict[str, Any],
) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": result.get("id", ""),
        "doc_id": result.get("metadata", {}).get("doc_id", ""),
        "source": result.get("metadata", {}).get("source", ""),
        "audience": result.get("metadata", {}).get("audience", ""),
        "score": round(float(result.get("score", 0.0)), 6),
        "matched_evidence": sorted(matched_evidence_ids(result["content"], benchmark_case)),
        "excerpt": result["content"].replace("\n", " ")[:240],
    }


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def render_text_report(payload: dict[str, Any]) -> str:
    lines = [
        "FPTU RAG BENCHMARK — 5 QUERIES / 7 METRICS",
        "=" * 88,
        f"Thời gian: {payload['generated_at']}",
        f"Embedding: {payload['embedding_backend']}",
        f"Chunking: RecursiveChunker(chunk_size={payload['chunk_size']})",
        f"Số chunks: {payload['chunk_count']}",
        f"Audience trong corpus: {', '.join(payload['audience_values'])}",
        f"Audience pre-filter: {'Có' if payload['audience_filter'] else 'Không'}",
        "",
        "ID\tRecall@1\tRecall@5\tMRR\tnDCG@5\tFull Evidence@5\tAgent Accuracy\tAudience Match",
    ]
    for query in payload["queries"]:
        metrics = query["metrics"]
        lines.append(
            "\t".join(
                [
                    query["id"],
                    percent(metrics["recall_at_1"]),
                    percent(metrics["recall_at_5"]),
                    percent(metrics["mrr"]),
                    percent(metrics["ndcg_at_5"]),
                    percent(metrics["full_evidence_at_5"]),
                    percent(metrics["faithfulness_agent_accuracy"]),
                    percent(metrics["audience_match_rate"]),
                ]
            )
        )

    average = payload["macro_average"]
    lines.extend(
        [
            "-" * 88,
            "MACRO\t"
            + "\t".join(
                percent(average[key])
                for key in (
                    "recall_at_1",
                    "recall_at_5",
                    "mrr",
                    "ndcg_at_5",
                    "full_evidence_at_5",
                    "faithfulness_agent_accuracy",
                    "audience_match_rate",
                )
            ),
            "",
            "ĐỊNH NGHĨA",
            "- Recall@k: tỷ lệ đơn vị bằng chứng gold xuất hiện trong top-k chunks.",
            "- MRR: nghịch đảo thứ hạng của chunk đầu tiên chứa bằng chứng gold.",
            "- nDCG@5: chất lượng thứ tự top-5; grade = số đơn vị bằng chứng trong chunk.",
            "- Full Evidence@5: 1 khi top-5 bao phủ toàn bộ bằng chứng bắt buộc, ngược lại 0.",
            "- Agent Accuracy: tỷ lệ nhóm từ khóa gold có mặt trong câu trả lời của Agent.",
            "- Audience Match: tỷ lệ top-5 có audience đúng đối tượng hoặc audience=all.",
            "",
        ]
    )

    if len(payload["audience_values"]) < 2:
        lines.extend(
            [
                "CẢNH BÁO AUDIENCE",
                "Corpus chỉ có một giá trị audience. Audience Match Rate vẫn được tính nhưng chưa",
                "đủ sức kiểm tra lỗi lấy nhầm tài liệu faculty/staff. Cần thêm nguồn khác audience",
                "hoặc một bộ distractor có provenance trước khi dùng chỉ số này để kết luận.",
                "",
            ]
        )

    lines.append("CHI TIẾT TỪNG QUERY")
    for query in payload["queries"]:
        lines.extend(["", f"{query['id']}. {query['query']}", f"Gold: {query['gold_answer']}"])
        if query["answer"] is not None:
            lines.append(f"Agent: {query['answer']}")
        for result in query["top_5"]:
            evidence = ",".join(result["matched_evidence"]) or "-"
            lines.append(
                f"  #{result['rank']} score={result['score']:.4f} "
                f"doc={result['doc_id']} audience={result['audience']} evidence={evidence}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    load_dotenv(override=False)
    args = parse_args()
    provider = args.provider or os.getenv("EMBEDDING_PROVIDER", "mock").strip().lower()
    cases = load_benchmark_cases()

    print(f"[1/4] Nạp corpus và chia chunk (size={args.chunk_size})")
    chunks = chunk_corpus(args.chunk_size)
    audience_values = sorted({chunk.metadata.get("audience", "") for chunk in chunks})

    print(f"[2/4] Khởi tạo embedding backend: {provider}")
    embedder, backend_name = select_embedder(provider)
    store = build_benchmark_store(chunks, embedder)

    print("[3/4] Chạy 5 benchmark queries")
    query_outputs: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    for index, case in enumerate(cases, 1):
        print(f"  {case['id']}: {case['query']}")
        if args.audience_filter:
            results = store.search_with_filter(
                case["query"],
                top_k=5,
                metadata_filter={"audience": case["expected_audience"]},
            )
        else:
            results = store.search(case["query"], top_k=5)

        answer = None if args.retrieval_only else generate_agent_answer(case["query"], results)
        metric_answer = answer or ""
        metrics: dict[str, float | None] = evaluate_query(results, chunks, case, metric_answer)
        if args.retrieval_only:
            metrics["faithfulness_agent_accuracy"] = None
        else:
            metric_rows.append({key: float(value) for key, value in metrics.items()})

        query_outputs.append(
            {
                "id": case["id"],
                "query": case["query"],
                "gold_answer": case["gold_answer"],
                "answer": answer,
                "metrics": metrics,
                "top_5": [
                    serialize_result(rank, result, case)
                    for rank, result in enumerate(results, 1)
                ],
            }
        )

    if args.retrieval_only:
        retrieval_rows = [
            {key: float(value) for key, value in output["metrics"].items() if value is not None}
            for output in query_outputs
        ]
        average = macro_average(retrieval_rows)
        average["faithfulness_agent_accuracy"] = None
    else:
        average = macro_average(metric_rows)

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "embedding_provider": provider,
        "embedding_backend": backend_name,
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "audience_values": audience_values,
        "audience_filter": args.audience_filter,
        "retrieval_only": args.retrieval_only,
        "macro_average": average,
        "queries": query_outputs,
    }

    print("[4/4] Ghi kết quả")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_text_report(payload), encoding="utf-8")
    args.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Báo cáo: {args.output}")
    print(f"  Dữ liệu JSON: {args.json_output}")
    print("\nMacro average:")
    for metric, value in average.items():
        print(f"  {metric}: {percent(value)}")


if __name__ == "__main__":
    main()

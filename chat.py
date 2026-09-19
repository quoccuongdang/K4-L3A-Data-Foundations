
"""
FPTU HCM Student Assistant — Interactive RAG Chat
===================================================
Hoi dap ve quy che, hoc phi, OJT, hoc bong, ... cua DH FPT HCM.

Chay:  python chat.py
Thoat: go 'exit' hoac 'quit' hoac nhan Ctrl+C
"""
from __future__ import annotations

import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=False)

from src import Document, EmbeddingStore, RecursiveChunker
from src.embeddings import GeminiEmbedder, _mock_embed

DATA_DIR = Path("data/university")


# ── Helpers ──────────────────────────────────────────────────────────

def load_docs() -> list[Document]:
    docs = []
    for f in sorted(DATA_DIR.glob("*.md")):
        if f.name == "CORPUS_MANIFEST.md":
            continue
        text = f.read_text(encoding="utf-8")
        meta = {"source": f.name}
        m = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
        if m:
            for line in m.group(1).split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"')
        content = re.sub(r"^---\n.*?\n---\n*", "", text, flags=re.DOTALL).strip()
        docs.append(Document(id=f.stem, content=content, metadata=meta))
    return docs


import json
import pickle
import hashlib

CACHE_FILE = DATA_DIR / ".embeddings_cache.pkl"


def get_cached_embedder(base_embedder):
    """Wrap embedder with persistent disk cache to avoid redundant API calls."""
    cache: dict[str, list[float]] = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    def cached_embed(text: str) -> list[float]:
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in cache:
            return cache[key]
        for attempt in range(5):
            try:
                vec = base_embedder(text)
                cache[key] = vec
                if len(cache) % 10 == 0:
                    save_cache()
                time.sleep(0.55)  # Pace to ~70 req/min to stay safely below 100 req/min limit
                return vec
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print(f"\n    [!] Chạm giới hạn Gemini Free Tier (100 req/phút). Đang đợi 12 giây...", flush=True)
                    save_cache()
                    time.sleep(12)
                else:
                    raise e
        raise RuntimeError("Khong the embedding sau 5 lan thu lai.")

    def save_cache():
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"[Warning] Khong the luu cache: {e}")

    cached_embed.save_cache = save_cache
    cached_embed._backend_name = getattr(base_embedder, "_backend_name", "cached")
    return cached_embed


def build_store(docs: list[Document], embedder) -> EmbeddingStore:
    chunker = RecursiveChunker(chunk_size=600)
    chunked: list[Document] = []
    for doc in docs:
        for i, chunk in enumerate(chunker.chunk(doc.content)):
            chunked.append(Document(
                id=f"{doc.id}_c{i}",
                content=chunk,
                metadata={**doc.metadata, "doc_id": doc.id, "chunk_idx": str(i)},
            ))
    store = EmbeddingStore(collection_name="fptu_chat", embedding_fn=embedder)
    
    total = len(chunked)
    print(f"[*] Dang embedding {total} chunks (co luu cache)...")
    for idx, doc in enumerate(chunked, 1):
        if idx % 25 == 0 or idx == total:
            print(f"    -> Da xu ly {idx}/{total} chunks...")
        store.add_documents([doc])
    
    if hasattr(embedder, "save_cache"):
        embedder.save_cache()
    return store


def call_gemini_llm(prompt: str) -> str:
    """Call Gemini API with automatic model fallback."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    configured_model = os.getenv("GEMINI_CHAT_MODEL", "").strip()
    models_to_try = [
        model
        for model in (
            configured_model,
            "gemini-3.5-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
        )
        if model
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Tat ca cac model Gemini deu gap loi: {last_error}")


def answer_question_with_sources(
    store: EmbeddingStore,
    question: str,
    top_k: int = 4,
) -> tuple[str, list[dict[str, str]]]:
    """Answer a question and return source metadata for web or CLI clients."""
    results = store.search(question, top_k=top_k)

    if not results:
        return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.", []

    # Build context
    context_parts = []
    sources: dict[str, dict[str, str]] = {}
    for i, r in enumerate(results, 1):
        src = r["metadata"].get("source", "?")
        cat = r["metadata"].get("category", "?")
        score = r.get("score", 0.0)
        context_parts.append(
            f"--- TAI LIEU {i} (Nguon: {src}, Danh muc: {cat}, Score: {score:.3f}) ---\n"
            f"{r['content']}"
        )
        sources[src] = {
            "file": src,
            "title": r["metadata"].get("title", src),
            "url": r["metadata"].get("source_url", ""),
            "category": cat,
        }

    context = "\n\n".join(context_parts)

    prompt = f"""Ban la tro ly tu van thong minh cua Truong Dai hoc FPT (FPTU) - campus TP. Ho Chi Minh.
Nhiem vu cua ban la tra loi cau hoi cua sinh vien mot cach chinh xac, ro rang va than thien dua tren cac thong tin duoc cung cap ben duoi.

Quy tac:
1. Chi tra loi dua tren cac thong tin trong phan THONG TIN THAM KHAO.
2. Neu thong tin khong du de tra loi day du, hay tra loi nhung gi co the va noi ro thong tin nao con thieu.
3. Trinh bay co danh sach gach dau dong hoac bang bieu neu can de sinh vien de theo doi.
4. Cuoi cau tra loi, neu co lien he phong ban (email/so dien thoai/phong) phu hop thi goi y cho sinh vien.

=== THONG TIN THAM KHAO ===
{context}

=== CAU HOI CUA SINH VIEN ===
{question}

=== CAU TRA LOI ==="""

    answer = call_gemini_llm(prompt)
    return answer, list(sources.values())


def answer_question(store: EmbeddingStore, question: str, top_k: int = 4) -> str:
    """Backward-compatible text response used by the command-line client."""
    answer, sources = answer_question_with_sources(store, question, top_k)
    source_list = ", ".join(f"`{source['file']}`" for source in sources)
    if not source_list:
        return answer
    return f"{answer}\n\n📌 **Tài liệu tham khảo:** {source_list}"


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  🎓 FPTU HCM Student Assistant (RAG Chatbot)")
    print("  Co so tri thuc: Quy che dao tao, Hoc phi, OJT, Hoc bong...")
    print("=" * 65)

    # Setup embedder
    print("\n[*] Dang khoi tao embedding backend...")
    try:
        raw_embedder = GeminiEmbedder()
        embedder = get_cached_embedder(raw_embedder)
        print(f"    -> Su dung Gemini Embedding voi bo nho dem Disk Cache")
    except Exception as e:
        print(f"    -> Gemini Embedding khong kha dung ({e}), fallback sang mock embedder")
        embedder = get_cached_embedder(_mock_embed)

    # Load & chunk documents
    print("[*] Dang tai tai lieu...")
    docs = load_docs()
    print(f"    -> Da tai {len(docs)} tai lieu tu data/university/")

    store = build_store(docs, embedder)
    print(f"    -> Hoan tat! Tong cong {store.get_collection_size()} chunks san sang trong Vector Store.")

    # Check LLM
    print("[*] Kiem tra ket noi Gemini LLM...")
    try:
        _ = call_gemini_llm("Ping")
        print("    -> Gemini LLM (gemini-3.5-flash) san sang!")
    except Exception as e:
        print(f"    -> [Loi LLM]: {e}")
        return

    # Check if a single query was passed via command line
    if len(sys.argv) > 1:
        single_query = " ".join(sys.argv[1:]).strip()
        print(f"\n[?] Cau hoi: {single_query}")
        print("-" * 65)
        ans = answer_question(store, single_query)
        print(f"\n{ans}\n")
        return

    # Interactive Loop
    print("\n" + "=" * 65)
    print("  San sang tro chuyen! Go cau hoi cua ban hoac 'exit' de thoat.")
    print("=" * 65)

    while True:
        try:
            print()
            question = input("👨‍🎓 Sinh vien: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTam biet ban!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "thoat", "bye"):
            print("\nChuc ban hoc tap tot tai FPTU! Tam biet!")
            break

        print("\n⏳ Dang truy xuat tri thuc va tao cau tra loi...\n")
        try:
            answer = answer_question(store, question)
            print(f"🤖 Tro ly FPTU:\n{answer}")
        except Exception as e:
            print(f"❌ [Loi]: {e}")


if __name__ == "__main__":
    main()

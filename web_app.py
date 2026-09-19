"""FastAPI bridge between the React interface and the existing FPTU RAG pipeline."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chat import (
    answer_question_with_sources,
    build_store,
    get_cached_embedder,
    load_docs,
)
from src.embeddings import GeminiEmbedder, _mock_embed

load_dotenv(override=False)


class Source(BaseModel):
    file: str
    title: str
    url: str = ""
    category: str = ""


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class StatusResponse(BaseModel):
    state: Literal["loading", "ready", "error"]
    chunks: int = 0
    documents: int = 0
    backend: str = ""
    message: str = ""


class RagRuntime:
    def __init__(self) -> None:
        self.state: Literal["loading", "ready", "error"] = "loading"
        self.store = None
        self.documents = 0
        self.backend = ""
        self.message = "Đang nạp kho tri thức…"
        self.lock = Lock()

    def initialize(self) -> None:
        try:
            docs = load_docs()
            try:
                embedder = get_cached_embedder(GeminiEmbedder())
            except Exception:
                embedder = get_cached_embedder(_mock_embed)

            store = build_store(docs, embedder)
            self.store = store
            self.documents = len(docs)
            self.backend = getattr(embedder, "_backend_name", "embedding")
            self.state = "ready"
            self.message = "Sẵn sàng trả lời"
        except Exception as exc:  # surfaced through /api/health
            self.state = "error"
            self.message = f"Không thể khởi tạo kho tri thức: {exc}"


runtime = RagRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialization = asyncio.create_task(asyncio.to_thread(runtime.initialize))
    yield
    if not initialization.done():
        initialization.cancel()


app = FastAPI(
    title="FPTU HCM Student Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=StatusResponse)
def health() -> StatusResponse:
    chunk_count = runtime.store.get_collection_size() if runtime.store else 0
    return StatusResponse(
        state=runtime.state,
        chunks=chunk_count,
        documents=runtime.documents,
        backend=runtime.backend,
        message=runtime.message,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if len(question) < 2:
        raise HTTPException(status_code=422, detail="Câu hỏi quá ngắn.")
    if runtime.state != "ready" or runtime.store is None:
        raise HTTPException(status_code=503, detail=runtime.message)

    def generate() -> tuple[str, list[dict[str, str]]]:
        with runtime.lock:
            return answer_question_with_sources(runtime.store, question)

    try:
        answer, sources = await asyncio.to_thread(generate)
        return ChatResponse(answer=answer, sources=[Source(**source) for source in sources])
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Không thể tạo câu trả lời lúc này: {exc}",
        ) from exc


# In production, FastAPI also serves the Vite build. During development Vite proxies /api.
DIST_DIR = Path(__file__).parent / "frontend" / "dist"
if DIST_DIR.exists():
    assets_dir = DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        candidate = (DIST_DIR / full_path).resolve()
        if candidate.is_file() and DIST_DIR.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)

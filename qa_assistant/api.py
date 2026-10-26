"""FastAPI web server: JSON API plus the single-page web UI.

Run:  uvicorn qa_assistant.api:app --reload
"""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qa_assistant.answerers import AnswerError
from qa_assistant.config import PROJECT_ROOT
from qa_assistant.pipeline import QAAssistant

STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(title="Context-Aware Q&A Assistant", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_assistant() -> QAAssistant:
    return QAAssistant()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    qa = get_assistant()
    return {
        "status": "ok",
        "mode": qa.mode,
        "model": qa.model,
        "chunks": len(qa.chunks),
        "documents": sorted({c.source for c in qa.chunks}),
    }


@app.get("/api/documents")
def documents() -> list[dict]:
    qa = get_assistant()
    docs: dict[str, dict] = {}
    for c in qa.chunks:
        docs.setdefault(c.source, {"source": c.source, "title": c.title, "sections": []})
        docs[c.source]["sections"].append({"id": c.id, "section": c.section})
    return list(docs.values())


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    try:
        return get_assistant().ask(request.question).to_dict()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except AnswerError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

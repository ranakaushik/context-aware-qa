"""The question-answering pipeline: validate -> retrieve -> answer -> verify citations."""

import time
from dataclasses import asdict, dataclass, field

import httpx
from google.genai import errors as genai_errors

from qa_assistant.answerers import (
    AnswerError,
    Draft,
    ExtractiveAnswerer,
    GeminiAnswerer,
    not_found,
)
from qa_assistant.config import Settings, get_settings, resolve_mode
from qa_assistant.knowledge_base import load_knowledge_base
from qa_assistant.retriever import BM25Retriever, RetrievedChunk
from qa_assistant.vector_retriever import GeminiEmbedder, VectorRetriever, fuse


@dataclass
class Source:
    id: str
    source: str
    section: str
    score: float
    text: str


@dataclass
class Answer:
    question: str
    status: str  # "answered" | "partial" | "not_found"
    answer: str
    citations: list[Source]
    missing_information: str
    retrieved: list[Source]
    mode: str
    latency_ms: int
    notes: list[str] = field(default_factory=list)

    @property
    def is_answered(self) -> bool:
        return self.status != "not_found"

    def to_dict(self) -> dict:
        return asdict(self)


def to_source(r: RetrievedChunk) -> Source:
    return Source(r.chunk.id, r.chunk.source, r.chunk.section, round(r.score, 3), r.chunk.text)


class QAAssistant:
    def __init__(self, settings: Settings | None = None, answerer=None, vector: VectorRetriever | None = None):
        """An injected answerer (as in the tests) gets keyword-only retrieval unless a vector retriever
        is injected too, so no embedding API calls happen behind the caller's back."""
        self.settings = settings or get_settings()
        self.chunks = load_knowledge_base(self.settings.kb_dir)
        self.retriever = BM25Retriever(self.chunks)
        if answerer is None:
            mode = resolve_mode(self.settings.mode)
            if mode == "gemini":
                answerer = GeminiAnswerer(self.settings.gemini_model)
                if self.settings.retrieval == "hybrid":
                    vector = VectorRetriever(
                        self.chunks,
                        GeminiEmbedder(self.settings.embedding_model, client=answerer.client),
                        cache_dir=self.settings.cache_dir,
                        min_similarity=self.settings.min_similarity,
                    )
            else:
                answerer = ExtractiveAnswerer(self.retriever.idf)
        self.answerer = answerer
        self.vector = vector

    @property
    def mode(self) -> str:
        return self.answerer.name

    @property
    def model(self) -> str | None:
        return self.settings.gemini_model if self.mode == "gemini" else None

    @property
    def retrieval(self) -> str:
        return "hybrid" if self.vector else "keyword"

    def retrieve(self, question: str, search_query: str) -> tuple[list[RetrievedChunk], list[str]]:
        """BM25 on the spelling-corrected query, plus (in hybrid mode) embedding search on the question as
        typed, merged by rank. Returns the chunks and any notes for the user."""
        top_k = self.settings.top_k
        keyword = self.retriever.search(search_query, top_k)
        if self.vector is None:
            return keyword, []
        try:
            semantic = self.vector.search(question, top_k)
        except (genai_errors.APIError, httpx.HTTPError, OSError):
            # Embeddings are an improvement, not a requirement: answer from keyword search alone.
            return keyword, ["Semantic search was unavailable, so only keyword search was used."]
        return fuse([keyword, semantic], top_k), []

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if len(question) > self.settings.max_question_chars:
            raise ValueError(f"Question is longer than {self.settings.max_question_chars} characters.")

        start = time.perf_counter()
        search_query, corrections = self.retriever.correct_spelling(question)
        retrieved, retrieval_notes = self.retrieve(question, search_query)
        if retrieved:
            # Gemini copes with typos itself and sees the original wording; the offline answerer needs
            # the corrected words.
            answer_question = search_query if self.mode == "offline" else question
            draft = self.answerer.answer(answer_question, retrieved)
        else:
            # No section shares a term with the question or is close in meaning: skip the model call.
            draft = not_found("No knowledge-base section matches the question's terms or meaning.")

        answer = self._finalize(question, draft, retrieved, start)
        answer.notes.extend(retrieval_notes)
        if corrections:
            answer.notes.insert(0, "Searched for " + ", ".join(f"'{new}' (you typed '{old}')" for old, new in corrections))
        return answer

    def _finalize(
        self, question: str, draft: Draft, retrieved: list[RetrievedChunk], start: float
    ) -> Answer:
        notes = []
        by_id = {r.chunk.id: r for r in retrieved}
        # Keep only citations that point at excerpts we actually supplied.
        citations = [to_source(by_id[c]) for c in dict.fromkeys(draft.citations) if c in by_id]
        dropped = [c for c in draft.citations if c not in by_id]
        if dropped:
            notes.append(f"Dropped citations not present in the retrieved context: {dropped}")
        if draft.status != "not_found" and not citations:
            notes.append("Answer has no valid citation; treat it with caution.")

        return Answer(
            question=question,
            status=draft.status,
            answer=draft.answer,
            citations=citations,
            missing_information=draft.missing_information,
            retrieved=[to_source(r) for r in retrieved],
            mode=self.mode,
            latency_ms=int((time.perf_counter() - start) * 1000),
            notes=notes,
        )


__all__ = ["Answer", "AnswerError", "QAAssistant", "Source"]

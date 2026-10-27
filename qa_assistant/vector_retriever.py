"""Semantic retrieval with Gemini embeddings, and fusion with the BM25 keyword ranking.

Vectors are small (74 sections x 768 numbers), so they live in memory and are searched with a plain
dot product; no vector database is needed. Section embeddings are cached on disk, keyed by model and
section text, so a restart does not re-embed unchanged sections.
"""

import hashlib
import json
import math
from pathlib import Path

from google import genai
from google.genai import types as genai_types

from qa_assistant.knowledge_base import Chunk
from qa_assistant.retriever import RetrievedChunk

DIMENSIONS = 768
BATCH_SIZE = 100  # the most texts the embedding API accepts per request
RRF_K = 60  # standard Reciprocal Rank Fusion constant; damps the gap between ranks 1 and 2


def normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / length for v in vector]


def chunk_text(chunk: Chunk) -> str:
    return f"{chunk.title} — {chunk.section}\n{chunk.text}"


class GeminiEmbedder:
    def __init__(self, model: str, client: genai.Client | None = None):
        self.model = model
        self.client = client or genai.Client()

    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            response = self.client.models.embed_content(
                model=self.model,
                contents=texts[start : start + BATCH_SIZE],
                config=genai_types.EmbedContentConfig(task_type=task_type, output_dimensionality=DIMENSIONS),
            )
            # Embeddings truncated below the model's full size are not unit length; cosine needs them to be.
            vectors.extend(normalize(e.values) for e in response.embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], "RETRIEVAL_QUERY")[0]


class VectorRetriever:
    """Cosine-similarity search over section embeddings.

    The index is built on the first search rather than at startup, so a failed embedding call does not
    stop the app from starting, and is retried on the next question.
    """

    def __init__(self, chunks: list[Chunk], embedder, cache_dir: Path | None = None, min_similarity: float = 0.6):
        self.chunks = chunks
        self.embedder = embedder
        # Questions unrelated to company policy ("capital of France") score about 0.5 with
        # gemini-embedding-001; policy questions score 0.65 or more.
        self.min_similarity = min_similarity
        self.cache_file = cache_dir / f"embeddings-{embedder.model}-{DIMENSIONS}.json" if cache_dir else None
        self.vectors: list[list[float]] | None = None

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def load_cache(self) -> dict[str, list[float]]:
        try:
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, AttributeError):
            return {}

    def save_cache(self, cache: dict[str, list[float]]) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(cache), encoding="utf-8")
        except (OSError, AttributeError):
            pass  # read-only disk or no cache configured: the index still works, it is just rebuilt next start

    def build_index(self) -> None:
        texts = [chunk_text(c) for c in self.chunks]
        cache = self.load_cache()
        missing = [t for t in dict.fromkeys(texts) if self.key(t) not in cache]
        if missing:
            cache.update(zip(map(self.key, missing), self.embedder.embed_documents(missing)))
            # Keep only current sections so edited or deleted ones do not pile up in the cache.
            cache = {self.key(t): cache[self.key(t)] for t in texts}
            self.save_cache(cache)
        self.vectors = [cache[self.key(t)] for t in texts]

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.vectors is None:
            self.build_index()
        q = self.embedder.embed_query(query)
        scored = [RetrievedChunk(c, sum(a * b for a, b in zip(q, v))) for c, v in zip(self.chunks, self.vectors)]
        scored = [r for r in scored if r.score >= self.min_similarity]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def fuse(rankings: list[list[RetrievedChunk]], top_k: int) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: a section's score is the sum of 1 / (RRF_K + rank) over the rankings it
    appears in. Uses ranks only, so BM25 scores and cosine similarities need no common scale."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, r in enumerate(ranking, start=1):
            scores[r.chunk.id] = scores.get(r.chunk.id, 0.0) + 1 / (RRF_K + rank)
            chunks[r.chunk.id] = r.chunk
    best = sorted(scores, key=scores.get, reverse=True)[:top_k]
    # Scaled so a section ranked first by both searches scores 1.0.
    top_possible = len(rankings) / (RRF_K + 1)
    return [RetrievedChunk(chunks[i], scores[i] / top_possible) for i in best]

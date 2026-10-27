import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import qa_assistant
from qa_assistant import api
from qa_assistant.answerers import GeminiAnswerer, to_sentences
from qa_assistant.config import get_settings
from qa_assistant.knowledge_base import load_knowledge_base
from qa_assistant.pipeline import QAAssistant
from qa_assistant.prompts import ANSWER_SCHEMA, SYSTEM_PROMPT
from qa_assistant.retriever import BM25Retriever, RetrievedChunk, stem, tokenize
from qa_assistant.vector_retriever import VectorRetriever, chunk_text, fuse


@pytest.fixture(scope="module")
def offline_qa():
    return QAAssistant(get_settings(mode="offline"))


# --- knowledge base and retrieval -------------------------------------------------------------

def test_knowledge_base_is_chunked_by_section():
    chunks = load_knowledge_base(get_settings().kb_dir)
    ids = {c.id for c in chunks}
    assert "leave_policy#carry-forward-of-annual-leave" in ids
    assert len(ids) == len(chunks), "chunk ids must be unique"


def test_stemmer_matches_plurals_and_past_tense():
    assert stem("days") == stem("day")
    assert stem("carried") == stem("carry")


def test_band_letters_survive_tokenization():
    assert "banda" in tokenize("Band A employees") and "bandc" in tokenize("band C")


def test_markdown_table_rows_become_sentences():
    text = "| Band | Train class |\n|---|---|\n| Band A | AC 2-tier |"
    assert to_sentences(text) == ["Band A: Train class AC 2-tier."]


def test_offline_answers_from_band_table(offline_qa):
    answer = offline_qa.ask("Which train class can a Band A employee book?")
    assert answer.answer.startswith("Band A:")
    assert "Train class AC 2-tier" in answer.answer


def test_retriever_ranks_relevant_section_first():
    retriever = BM25Retriever(load_knowledge_base(get_settings().kb_dir))
    top = retriever.search("Can employees carry forward more than 10 days of leave?", top_k=3)
    assert top[0].chunk.id == "leave_policy#carry-forward-of-annual-leave"


def test_retriever_returns_nothing_for_unrelated_question():
    retriever = BM25Retriever(load_knowledge_base(get_settings().kb_dir))
    assert retriever.search("Who won the cricket World Cup?") == []


# --- offline pipeline -------------------------------------------------------------------------

def test_offline_answers_from_policy(offline_qa):
    answer = offline_qa.ask("Can employees carry forward more than 10 days of leave?")
    assert answer.status == "answered"
    assert "10 days" in answer.answer
    assert answer.citations[0].source == "leave_policy.md"


@pytest.mark.parametrize("question", [
    "What is the company's dress code?",
    "How many stock options do new employees receive?",
    "What is the capital of France?",
])
def test_offline_declines_when_answer_missing(offline_qa, question):
    answer = offline_qa.ask(question)
    assert answer.status == "not_found"
    assert answer.citations == []


def test_misspelled_question_is_corrected(offline_qa):
    answer = offline_qa.ask("per dem for internatioal trip")
    assert answer.status == "answered"
    assert answer.citations[0].id == "expense_policy#international-travel-per-diem"
    assert "you typed 'internatioal'" in answer.notes[0]


def test_spelling_correction_does_not_invent_coverage(offline_qa):
    # "relocation" is a real word the policies don't cover; it must not become "location".
    assert offline_qa.ask("Is there a relocation allowance when I transfer to another city?").status == "not_found"


@pytest.mark.parametrize("question", ["International Travel: Per Diem", "DA for international travel"])
def test_topic_question_returns_whole_table(offline_qa, question):
    answer = offline_qa.ask(question)
    assert answer.citations[0].id == "expense_policy#international-travel-per-diem"
    assert all(group in answer.answer for group in ("Group 1:", "Group 2:", "Group 3:"))


def test_empty_question_is_rejected(offline_qa):
    with pytest.raises(ValueError):
        offline_qa.ask("   ")


# --- Gemini pipeline with a fake Gemini client --------------------------------------------------

class FakeGeminiModels:
    def __init__(self, payload, finish_reason="STOP"):
        self.payload = payload
        self.finish_reason = finish_reason
        self.last_request = None

    def generate_content(self, **kwargs):
        from google.genai import types
        self.last_request = kwargs
        candidate = SimpleNamespace(finish_reason=types.FinishReason(self.finish_reason))
        return SimpleNamespace(candidates=[candidate], text=json.dumps(self.payload))


def fake_gemini(payload, finish_reason="STOP"):
    models = FakeGeminiModels(payload, finish_reason)
    return SimpleNamespace(models=models), models


def test_gemini_request_uses_same_prompt_and_schema():
    client, models = fake_gemini({
        "status": "answered",
        "answer": "Band A gets ₹13,500 per day in Group 1 countries.",
        "citations": ["expense_policy#international-travel-per-diem"],
        "missing_information": "",
    })
    qa = QAAssistant(get_settings(), answerer=GeminiAnswerer("gemini-flash-latest", client=client))
    answer = qa.ask("What is the per diem for a Band A employee travelling to a Group 1 country?")

    request = models.last_request
    assert request["model"] == "gemini-flash-latest"
    assert request["config"].system_instruction == SYSTEM_PROMPT
    assert request["config"].response_json_schema == ANSWER_SCHEMA
    assert request["config"].response_mime_type == "application/json"
    assert "<context>" in request["contents"]
    assert answer.mode == "gemini" and answer.status == "answered"
    assert [c.id for c in answer.citations] == ["expense_policy#international-travel-per-diem"]


def test_citations_outside_retrieved_context_are_dropped():
    client, _ = fake_gemini({
        "status": "answered",
        "answer": "No. At most 10 days can be carried forward.",
        "citations": ["leave_policy#carry-forward-of-annual-leave", "made-up#chunk"],
        "missing_information": "",
    })
    qa = QAAssistant(get_settings(), answerer=GeminiAnswerer("gemini-flash-latest", client=client))
    answer = qa.ask("Can employees carry forward more than 10 days of leave?")
    assert [c.id for c in answer.citations] == ["leave_policy#carry-forward-of-annual-leave"]
    assert answer.notes


def test_gemini_not_called_when_nothing_retrieved():
    client, models = fake_gemini({})
    qa = QAAssistant(get_settings(), answerer=GeminiAnswerer("gemini-flash-latest", client=client))
    assert qa.ask("Who won the cricket World Cup?").status == "not_found"
    assert models.last_request is None


def test_gemini_safety_block_becomes_not_found():
    client, _ = fake_gemini({}, finish_reason="SAFETY")
    qa = QAAssistant(get_settings(), answerer=GeminiAnswerer("gemini-flash-latest", client=client))
    assert qa.ask("How many sick days do I get?").status == "not_found"


def test_auto_mode_uses_gemini_only_when_key_is_set(monkeypatch):
    from qa_assistant.config import resolve_mode
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert resolve_mode("auto") == "gemini"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert resolve_mode("auto") == "offline"


# --- hybrid retrieval with a fake embedder -----------------------------------------------------

class FakeEmbedder:
    """One-hot vector per section; a query is embedded as the vector of the section mapped to it."""

    model = "fake-embedding"

    def __init__(self, chunks, query_targets=None, fail=False):
        self.index = {chunk_text(c): i for i, c in enumerate(chunks)}
        self.ids = [c.id for c in chunks]
        self.query_targets = query_targets or {}
        self.fail = fail
        self.document_calls = 0

    def one_hot(self, i):
        return [1.0 if j == i else 0.0 for j in range(len(self.ids))]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self.one_hot(self.index[t]) for t in texts]

    def embed_query(self, text):
        if self.fail:
            from google.genai import errors
            raise errors.ServerError(503, {"error": {"code": 503, "message": "high demand"}})
        target = self.query_targets.get(text)
        return self.one_hot(self.ids.index(target)) if target else [0.0] * len(self.ids)


def hybrid_qa(tmp_path, payload, **embedder_args):
    settings = get_settings()
    chunks = load_knowledge_base(settings.kb_dir)
    client, models = fake_gemini(payload)
    vector = VectorRetriever(chunks, FakeEmbedder(chunks, **embedder_args), cache_dir=tmp_path)
    return QAAssistant(settings, answerer=GeminiAnswerer("gemini-flash-latest", client=client), vector=vector), models


def test_fuse_prefers_sections_found_by_both_searches():
    chunks = load_knowledge_base(get_settings().kb_dir)
    a, b, c = (RetrievedChunk(chunk, 1.0) for chunk in chunks[:3])
    fused = fuse([[a, b], [c, b]], top_k=3)
    assert fused[0].chunk.id == b.chunk.id
    assert {r.chunk.id for r in fused} == {a.chunk.id, b.chunk.id, c.chunk.id}


def test_semantic_search_finds_section_keyword_search_misses(tmp_path):
    question = "Is my puppy welcome?"  # shares no term with the knowledge base
    target = "remote_work_policy#hybrid-work-model"
    qa, models = hybrid_qa(tmp_path, {
        "status": "not_found", "answer": "x", "citations": [], "missing_information": "pets",
    }, query_targets={question: target})
    assert qa.retriever.search(question) == []
    answer = qa.ask(question)
    assert [r.id for r in answer.retrieved] == [target]
    assert models.last_request is not None
    assert qa.retrieval == "hybrid"


def test_off_topic_question_still_skips_the_model_in_hybrid_mode(tmp_path):
    qa, models = hybrid_qa(tmp_path, {})
    assert qa.ask("Who won the cricket World Cup?").status == "not_found"
    assert models.last_request is None


def test_section_embeddings_are_cached(tmp_path):
    chunks = load_knowledge_base(get_settings().kb_dir)
    first = FakeEmbedder(chunks)
    VectorRetriever(chunks, first, cache_dir=tmp_path).search("leave")
    second = FakeEmbedder(chunks)
    VectorRetriever(chunks, second, cache_dir=tmp_path).search("leave")
    assert (first.document_calls, second.document_calls) == (1, 0)


def test_embedding_failure_falls_back_to_keyword_search(tmp_path):
    qa, models = hybrid_qa(tmp_path, {
        "status": "answered",
        "answer": "No. At most 10 days can be carried forward.",
        "citations": ["leave_policy#carry-forward-of-annual-leave"],
        "missing_information": "",
    }, fail=True)
    answer = qa.ask("Can employees carry forward more than 10 days of leave?")
    assert answer.status == "answered"
    assert answer.retrieved[0].id == "leave_policy#carry-forward-of-annual-leave"
    assert any("only keyword search" in n for n in answer.notes)


# --- HTTP API ---------------------------------------------------------------------------------

def test_api_ask_and_health(monkeypatch, offline_qa):
    monkeypatch.setattr(api, "get_assistant", lambda: offline_qa)
    client = TestClient(api.app)
    health = client.get("/api/health").json()
    assert health["mode"] == "offline"
    assert health["version"] == qa_assistant.__version__
    response = client.post("/api/ask", json={"question": "How much is the home office stipend?"})
    assert response.status_code == 200
    assert "₹25,000" in response.json()["answer"]
    assert client.post("/api/ask", json={"question": ""}).status_code == 422

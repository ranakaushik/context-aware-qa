import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from qa_assistant import api
from qa_assistant.answerers import GeminiAnswerer, to_sentences
from qa_assistant.config import get_settings
from qa_assistant.knowledge_base import load_knowledge_base
from qa_assistant.pipeline import QAAssistant
from qa_assistant.prompts import ANSWER_SCHEMA, SYSTEM_PROMPT
from qa_assistant.retriever import BM25Retriever, stem, tokenize


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


# --- HTTP API ---------------------------------------------------------------------------------

def test_api_ask_and_health(monkeypatch, offline_qa):
    monkeypatch.setattr(api, "get_assistant", lambda: offline_qa)
    client = TestClient(api.app)
    assert client.get("/api/health").json()["mode"] == "offline"
    response = client.post("/api/ask", json={"question": "How much is the home office stipend?"})
    assert response.status_code == 200
    assert "₹25,000" in response.json()["answer"]
    assert client.post("/api/ask", json={"question": ""}).status_code == 422

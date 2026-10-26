"""Answer generators: Google Gemini (grounded LLM answer) and an offline extractive fallback."""

import json
import re
from dataclasses import dataclass, field

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from qa_assistant.prompts import ANSWER_SCHEMA, NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_message
from qa_assistant.retriever import RetrievedChunk, term_groups, tokenize


@dataclass
class Draft:
    status: str  # "answered" | "partial" | "not_found"
    answer: str
    citations: list[str] = field(default_factory=list)
    missing_information: str = ""


class AnswerError(RuntimeError):
    """Raised when an answer could not be produced (API failure, malformed output)."""


def not_found(reason: str) -> Draft:
    return Draft(status="not_found", answer=NOT_FOUND_MESSAGE, missing_information=reason)


class GeminiAnswerer:
    """Sends the question plus retrieved excerpts to Gemini (google-genai SDK) and parses a structured
    JSON answer that must match ANSWER_SCHEMA."""

    name = "gemini"

    def __init__(self, model: str, client: genai.Client | None = None):
        self.model = model
        # genai.Client() reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
        self.client = client or genai.Client()

    def answer(self, question: str, retrieved: list[RetrievedChunk]) -> Draft:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=build_user_message(question, retrieved),
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_json_schema=ANSWER_SCHEMA,
                    max_output_tokens=16000,
                ),
            )
        except genai_errors.ClientError as e:
            # Google answers an invalid key with 400 "API key not valid", a missing permission with 401/403.
            if e.code in (401, 403) or "API key" in (e.message or ""):
                raise AnswerError("Gemini rejected the credentials. Check GEMINI_API_KEY.") from e
            if e.code == 404:
                raise AnswerError(
                    f"Gemini model '{self.model}' was not found. Set GEMINI_MODEL to a model from "
                    "`python -m qa_assistant.gemini_models`."
                ) from e
            if e.code == 429:
                raise AnswerError("Gemini rate limit or quota reached. Wait a moment and try again.") from e
            raise AnswerError(f"Gemini API error {e.code}: {e.message}") from e
        except genai_errors.APIError as e:
            raise AnswerError(f"Gemini API error {e.code}: {e.message}") from e
        except OSError as e:
            raise AnswerError("Could not reach the Gemini API. Check your network connection.") from e

        candidate = response.candidates[0] if response.candidates else None
        finish = candidate.finish_reason if candidate else None
        blocked = {
            genai_types.FinishReason.SAFETY,
            genai_types.FinishReason.PROHIBITED_CONTENT,
            genai_types.FinishReason.BLOCKLIST,
            genai_types.FinishReason.SPII,
        }
        if candidate is None or finish in blocked:
            return not_found("The model declined to answer this request.")
        if finish == genai_types.FinishReason.MAX_TOKENS:
            raise AnswerError("The answer was cut off (max output tokens reached).")

        try:
            data = json.loads(response.text or "")
        except json.JSONDecodeError as e:
            raise AnswerError("Gemini returned output that was not valid JSON.") from e
        return Draft(
            status=data["status"],
            answer=data["answer"],
            citations=list(data["citations"]),
            missing_information=data["missing_information"],
        )


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_units(text: str) -> list[tuple[str, int | None]]:
    """Split chunk text into (sentence, table number) pairs; table number is None for prose.

    Markdown table rows become sentences such as "Band A: Train class AC 2-tier; ...".
    """
    units, header, table = [], None, -1
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # separator row
            if header is None:
                header, table = cells, table + 1
            else:
                pairs = "; ".join(f"{h} {c}" for h, c in zip(header[1:], cells[1:]))
                units.append((f"{cells[0]}: {pairs}.", table))
        else:
            header = None
            units.extend((s, None) for s in SENTENCE_RE.split(line) if s)
    return units


def to_sentences(text: str) -> list[str]:
    return [sentence for sentence, _ in split_units(text)]


class ExtractiveAnswerer:
    """No-LLM fallback: returns the best-matching sentences, or "not found" when coverage is weak.

    It cannot reason or paraphrase, so it is a baseline and an offline demo mode, not a substitute
    for the LLM answerer.
    """

    name = "offline"

    def __init__(self, idf: dict[str, float], min_coverage: float = 0.65, max_sentences: int = 2):
        # Coverage is IDF-weighted: missing a rare word ("stock") matters more than a common one.
        self.idf = idf
        self.unseen_idf = max(idf.values(), default=1.0)
        self.min_coverage = min_coverage
        self.max_sentences = max_sentences

    @staticmethod
    def table_answer(chunk, units: list[tuple[str, int | None]], table: int) -> Draft:
        """The whole table, preceded by the sentence that introduces it."""
        rows = [u for u, t in units if t == table]
        first = next(i for i, (_, t) in enumerate(units) if t == table)
        intro = [units[first - 1][0]] if first > 0 and units[first - 1][1] is None else []
        return Draft(status="answered", answer=" ".join(intro + rows), citations=[chunk.id])

    def weight(self, group: set[str]) -> float:
        return max(self.idf.get(term, self.unseen_idf) for term in group)

    def coverage(self, groups: list[set[str]], terms: set[str]) -> float:
        # Numbers in a question are usually values to reason about ("a 4-hour flight",
        # "₹1,00,000"), not topic words, so an unmatched number does not count against coverage.
        groups = [g for g in groups if g & terms or not all(t.isdigit() for t in g)]
        if not groups:
            return 0.0
        return sum(self.weight(g) for g in groups if g & terms) / sum(self.weight(g) for g in groups)

    def answer(self, question: str, retrieved: list[RetrievedChunk]) -> Draft:
        groups = term_groups(question)
        if not retrieved or not groups:
            return not_found("No knowledge-base section matched the question.")

        top = retrieved[0].chunk
        top_terms = set(tokenize(f"{top.title} {top.section} {top.text}"))
        # A question word is covered if it or one of its synonyms appears ("vacation" -> "leave").
        coverage = self.coverage(groups, top_terms)
        if coverage < self.min_coverage:
            return not_found(
                f"The closest section ({top.id}) covers only {coverage:.0%} of the question's key terms (IDF-weighted)."
            )

        # A topic question ("International Travel: Per Diem", "per diem for international trip") that names
        # no band is answered with the table of the section whose heading it matches.
        band_terms = {"banda", "bandb", "bandc"}
        if not any(g & band_terms for g in groups):
            with_tables = [r for r in retrieved[:3] if any(t is not None for _, t in split_units(r.chunk.text))]
            scored = [(self.coverage(groups, set(tokenize(r.chunk.section))), -i, r) for i, r in enumerate(with_tables)]
            if scored:
                heading_coverage, _, best = max(scored, key=lambda x: x[:2])
                if heading_coverage >= self.min_coverage:
                    return self.table_answer(best.chunk, split_units(best.chunk.text), table=0)

        expanded_query = set().union(*groups)
        candidates = []
        for rank, r in enumerate(retrieved[:2]):
            for position, (sentence, table) in enumerate(split_units(r.chunk.text)):
                # IDF-weighted, so a rare match ("bandb") outranks several common ones ("travel").
                overlap = sum(self.idf.get(t, self.unseen_idf) for t in expanded_query & set(tokenize(sentence)))
                if overlap:
                    # Prefer sentences from the top-ranked chunk on ties.
                    candidates.append((overlap, -rank, -position, sentence.strip(), table, r.chunk))
        if not candidates:
            return not_found("No sentence in the matched sections relates to the question.")

        # A question about a specific band is answered from sentences or table rows that mention that band.
        asked_bands = {t for g in groups for t in g & band_terms}
        if asked_bands:
            about_band = [c for c in candidates if asked_bands & set(tokenize(c[3]))]
            candidates = about_band or candidates

        candidates.sort(key=lambda c: c[:3], reverse=True)
        _, _, _, _, table, chunk = candidates[0]
        if table is not None and not any(g & band_terms for g in groups):
            # The best match is a table row but no specific band was asked about: show the whole table.
            return self.table_answer(chunk, split_units(chunk.text), table)

        best = candidates[: self.max_sentences]
        best_parts = [(c[3], c[5].id) for c in best]
        return Draft(
            status="answered",
            answer=" ".join(text for text, _ in best_parts),
            citations=list(dict.fromkeys(cid for _, cid in best_parts)),
        )

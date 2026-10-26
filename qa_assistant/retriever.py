"""BM25 keyword retriever over knowledge-base chunks (pure Python, no external index)."""

import difflib
import math
import re
from collections import Counter
from dataclasses import dataclass

from qa_assistant.knowledge_base import Chunk

STOPWORDS = frozenset(
    """a about above after again all am an and any are as at be been before being below between both
    but by can could did do does doing down during each few for from further had has have having he her
    here hers him his how i if in into is it its itself just me more most my myself no nor not of off on
    once only or other our ours out over own same she should so some such than that the their theirs them
    then there these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours i'm i've get got may must also per much many""".split()
)

# Small synonym map so everyday wording reaches policy wording ("vacation" -> "leave").
SYNONYMS = {
    "vacation": ["leave", "annual"],
    "holiday": ["leave", "holidays"],
    "pto": ["leave", "annual"],
    "wfh": ["remote", "home"],
    "abroad": ["country"],
    "overseas": ["country", "international"],
    "reimburse": ["expense", "reimbursement"],
    "reimbursed": ["expense", "reimbursement"],
    "maternity": ["parental"],
    "paternity": ["parental"],
    "401k": ["provident", "fund"],
    "pension": ["provident", "gratuity"],
    "retirement": ["provident", "gratuity"],
    "pf": ["provident", "fund"],
    "epf": ["provident", "fund"],
    "mileage": ["km", "conveyance"],
    "petrol": ["vehicle", "km"],
    "fuel": ["vehicle", "km"],
    "deadline": ["within"],
    "trip": ["travel"],
    "trips": ["travel"],
    "hotel": ["stay"],
    "hotels": ["stay"],
    "accommodation": ["stay", "hotel"],
    "lodging": ["stay", "hotel"],
    "dem": ["diem"],
    "diam": ["diem"],
    "perdiem": ["diem"],
    "da": ["diem", "allowance"],
    "fly": ["flight"],
    "flying": ["flight"],
    "kilometre": ["km"],
    "kilometer": ["km"],
    "mediclaim": ["health", "insurance"],
    "diwali": ["festival", "holiday"],
    "laptop": ["device", "laptop"],
    "fired": ["termination"],
    "quit": ["resignation"],
    "resign": ["resignation"],
    "salary": ["salary", "salaries"],
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.:,][0-9]+)*")
# "Band A" would lose its letter to the stopword list, so glue grade letters on: "band a" -> "banda".
BAND_RE = re.compile(r"\bband\s+([a-z])\b")


def stem(word: str) -> str:
    """Very small suffix stripper; enough to match "days"/"day" or "carried"/"carry"."""
    if word.isdigit() or len(word) <= 3:
        return word
    for suffix, replacement in (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def tokenize(text: str, expand_synonyms: bool = False) -> list[str]:
    words = [w.replace(",", "") for w in TOKEN_RE.findall(BAND_RE.sub(r"band\1", text.lower()))]
    tokens = []
    for word in words:
        if word in STOPWORDS:
            continue
        tokens.append(stem(word))
        if expand_synonyms:
            tokens.extend(stem(s) for s in SYNONYMS.get(word, []))
    return tokens


def term_groups(text: str) -> list[set[str]]:
    """One set per content word: its stem plus the stems of its synonyms."""
    groups = []
    for word in (w.replace(",", "") for w in TOKEN_RE.findall(BAND_RE.sub(r"band\1", text.lower()))):
        if word not in STOPWORDS:
            groups.append({stem(word), *(stem(s) for s in SYNONYMS.get(word, []))})
    return groups


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        # The section heading is included twice so a matching heading counts more than body text.
        self.doc_tokens = [
            tokenize(f"{c.title} {c.section} {c.section} {c.text}") for c in chunks
        ]
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.avg_len = sum(len(t) for t in self.doc_tokens) / max(len(chunks), 1)
        doc_freq = Counter(term for tokens in self.doc_tokens for term in set(tokens))
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}
        self.heading_terms = [set(tokenize(c.section)) for c in chunks]
        # Unstemmed words of the knowledge base, used for spelling correction.
        self.vocabulary = sorted(
            {w for c in chunks for w in TOKEN_RE.findall(f"{c.title} {c.section} {c.text}".lower()) if w.isalpha()}
        )

    def score(self, query_terms: list[str], index: int) -> float:
        tf = self.term_freqs[index]
        length_norm = 1 - self.b + self.b * len(self.doc_tokens[index]) / self.avg_len
        total = 0.0
        for term in set(query_terms):
            freq = tf.get(term, 0)
            if freq:
                total += self.idf[term] * freq * (self.k1 + 1) / (freq + self.k1 * length_norm)
        return total

    def correct_spelling(self, query: str) -> tuple[str, list[tuple[str, str]]]:
        """Replace misspelled words with the closest knowledge-base word ("internatioal" -> "international").

        Only words of 5+ letters that are unknown to the index are touched, and the replacement must share
        the first letter, so real but uncovered words ("relocation") are not turned into covered ones.
        """
        corrections = []

        def fix(match: re.Match) -> str:
            word = match.group(0)
            lower = word.lower()
            if len(lower) < 5 or not lower.isalpha() or lower in STOPWORDS or lower in SYNONYMS:
                return word
            if stem(lower) in self.idf:
                return word
            close = difflib.get_close_matches(lower, self.vocabulary, n=1, cutoff=0.8)
            if close and close[0][0] == lower[0]:
                corrections.append((word, close[0]))
                return close[0]
            return word

        return re.sub(r"[A-Za-z]+", fix, query), corrections

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_terms = tokenize(query, expand_synonyms=True)
        plain_terms = set(tokenize(query))
        scored = []
        for i, c in enumerate(self.chunks):
            score = self.score(query_terms, i)
            # A question that is just a section heading ("International Travel: Per Diem") gets that section.
            if len(plain_terms) >= 2 and plain_terms <= self.heading_terms[i]:
                score *= 1.5
            scored.append(RetrievedChunk(c, score))
        scored = [r for r in scored if r.score > 0]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

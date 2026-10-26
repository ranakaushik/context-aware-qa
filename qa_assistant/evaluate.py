"""Runs the evaluation question set and writes JSON + Markdown reports to eval/results/.

Usage:  python -m qa_assistant.evaluate [--mode offline|gemini|auto]
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from qa_assistant.config import PROJECT_ROOT, get_settings
from qa_assistant.pipeline import QAAssistant

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def keyword_present(keyword: str, text: str) -> bool:
    # Word-boundary match so "no" does not match "not" or "know"; symbols like "5%" match literally.
    pattern = r"(?<![A-Za-z0-9])" + re.escape(keyword.lower()) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text.lower()) is not None


def grade(item: dict, answer) -> dict:
    retrieved_sources = [r.source for r in answer.retrieved]
    if item["category"] == "answerable":
        matched = [any(keyword_present(k, answer.answer) for k in group) for group in item["expected_keywords"]]
        passed = answer.status in ("answered", "partial") and all(matched)
        retrieval_hit = item["expected_source"] in retrieved_sources
        verdict = "pass" if passed else ("wrongly_declined" if answer.status == "not_found" else "wrong_answer")
    else:
        passed = answer.status == "not_found"
        retrieval_hit = None
        verdict = "pass" if passed else "hallucinated"
    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "status": answer.status,
        "answer": answer.answer,
        "citations": [c.id for c in answer.citations],
        "top_retrieved": [r.id for r in answer.retrieved[:3]],
        "retrieval_hit": retrieval_hit,
        "passed": passed,
        "verdict": verdict,
        "latency_ms": answer.latency_ms,
    }


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["category"] == "answerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]

    def pct(n, d):
        return round(100 * n / d, 1) if d else 0.0

    return {
        "total_questions": len(rows),
        "overall_accuracy_pct": pct(sum(r["passed"] for r in rows), len(rows)),
        "answerable_accuracy_pct": pct(sum(r["passed"] for r in answerable), len(answerable)),
        "unanswerable_decline_rate_pct": pct(sum(r["passed"] for r in unanswerable), len(unanswerable)),
        "hallucination_rate_pct": pct(sum(r["verdict"] == "hallucinated" for r in unanswerable), len(unanswerable)),
        "wrongly_declined": sum(r["verdict"] == "wrongly_declined" for r in answerable),
        "retrieval_hit_rate_pct": pct(sum(bool(r["retrieval_hit"]) for r in answerable), len(answerable)),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / max(len(rows), 1)),
    }


def to_markdown(mode: str, model: str, summary: dict, rows: list[dict]) -> str:
    lines = [
        f"# Evaluation results — mode `{mode}`",
        "",
        f"Run: {datetime.now():%Y-%m-%d %H:%M}  ",
        f"Model: `{model or 'n/a (offline extractive baseline)'}`",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    lines += [
        "",
        "| ID | Category | Question | Status | Verdict | Answer |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        answer = r["answer"].replace("|", "\\|").replace("\n", " ")
        mark = "✅" if r["passed"] else "❌"
        lines.append(
            f"| {r['id']} | {r['category']} | {r['question']} | {r['status']} | {mark} {r['verdict']} | {answer} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Q&A assistant.")
    parser.add_argument("--mode", choices=["auto", "gemini", "offline"], default=None)
    args = parser.parse_args()

    settings = get_settings(mode=args.mode)
    qa = QAAssistant(settings)
    items = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    rows = []
    for item in items:
        row = grade(item, qa.ask(item["question"]))
        rows.append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'}  {row['id']}  {row['status']:<10} {item['question']}")

    summary = summarize(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"results_{qa.mode}.json").write_text(
        json.dumps({"mode": qa.mode, "model": qa.model, "summary": summary, "results": rows}, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"results_{qa.mode}.md").write_text(
        to_markdown(qa.mode, qa.model, summary, rows), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, indent=2))
    print(f"Reports written to {RESULTS_DIR}/results_{qa.mode}.(json|md)")


if __name__ == "__main__":
    main()

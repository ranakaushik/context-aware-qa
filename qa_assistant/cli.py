"""Command-line interface.

Usage:
  python -m qa_assistant.cli "Can employees carry forward more than 10 days of leave?"
  python -m qa_assistant.cli            # interactive mode
  python -m qa_assistant.cli --mode offline --show-context "..."
"""

import argparse

from qa_assistant.answerers import AnswerError
from qa_assistant.config import get_settings
from qa_assistant.pipeline import Answer, QAAssistant

STATUS_LABELS = {"answered": "ANSWERED", "partial": "PARTIAL ANSWER", "not_found": "NOT IN KNOWLEDGE BASE"}


def print_answer(answer: Answer, show_context: bool) -> None:
    print(f"\n[{STATUS_LABELS[answer.status]}]  ({answer.mode} mode, {answer.latency_ms} ms)")
    print(answer.answer)
    if answer.missing_information:
        print(f"\nMissing: {answer.missing_information}")
    if answer.citations:
        print("\nSources:")
        for c in answer.citations:
            print(f"  - {c.source} › {c.section}")
    for note in answer.notes:
        print(f"Note: {note}")
    if show_context:
        print("\nRetrieved context:")
        for r in answer.retrieved:
            print(f"  [{r.score:6.2f}] {r.id}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the knowledge base a question.")
    parser.add_argument("question", nargs="*", help="question to ask (omit for interactive mode)")
    parser.add_argument("--mode", choices=["auto", "gemini", "offline"], default=None)
    parser.add_argument("--show-context", action="store_true", help="print the retrieved chunks")
    args = parser.parse_args()

    qa = QAAssistant(get_settings(mode=args.mode))
    print(f"Loaded {len(qa.chunks)} knowledge-base sections. Mode: {qa.mode}")

    def run(question: str) -> None:
        try:
            print_answer(qa.ask(question), args.show_context)
        except (ValueError, AnswerError) as e:
            print(f"Error: {e}\n")

    if args.question:
        run(" ".join(args.question))
        return

    print("Type a question (or 'exit' to quit).")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            run(question)


if __name__ == "__main__":
    main()

"""Context-Aware Q&A Assistant: answers questions using only a supplied knowledge base."""

__version__ = "1.1.0"

from qa_assistant.pipeline import Answer, QAAssistant  # noqa: E402

__all__ = ["Answer", "QAAssistant", "__version__"]

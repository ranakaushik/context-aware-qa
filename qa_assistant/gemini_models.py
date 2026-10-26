"""Lists the Gemini models your API key can use for answering (they support generateContent).

Usage:  python -m qa_assistant.gemini_models
Then set GEMINI_MODEL in .env to one of the printed names.
"""

from google import genai
from google.genai import errors

from qa_assistant import config  # noqa: F401  (loads .env)
from qa_assistant.config import has_gemini_credentials


def main() -> None:
    if not has_gemini_credentials():
        raise SystemExit("Set GEMINI_API_KEY in .env first (get a key at https://aistudio.google.com/apikey).")
    # Keep a reference to the client: the model list is paged lazily and needs the client open.
    client = genai.Client()
    try:
        models = [m for m in client.models.list() if "generateContent" in (m.supported_actions or [])]
    except errors.APIError as e:
        raise SystemExit(f"Gemini API error {e.code}: {e.message}")
    for m in sorted(models, key=lambda m: m.name):
        print(f"{m.name.removeprefix('models/'):45} {m.display_name or ''}")


if __name__ == "__main__":
    main()

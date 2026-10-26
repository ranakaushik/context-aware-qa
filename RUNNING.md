# Running the Application — Step by Step

## 0. Prerequisites
- Python **3.10 or newer** (`python3 --version`)
- Optional but recommended: a Google Gemini API key from <https://aistudio.google.com/apikey> (without it the app runs in offline mode)

## 1. Open the project folder
```bash
cd context-aware-qa
```

## 2. Create and activate a virtual environment
macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Configure the Gemini API key
```bash
cp .env.example .env
```
Create a key at <https://aistudio.google.com/apikey>, then open `.env` and set:
```
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-flash-latest
```
To see which Gemini models your key can use, and pin one of them:
```bash
python -m qa_assistant.gemini_models
```
Copy a name from the list into `GEMINI_MODEL` (for example `gemini-3-flash-preview` or `gemini-3.1-pro-preview`). The default `gemini-flash-latest` always follows Google's current Flash model.

**Mode** — `QA_MODE` in `.env`:

| `QA_MODE` | Answers with |
|---|---|
| `auto` (default) | Gemini if `GEMINI_API_KEY` is set, else offline |
| `gemini` | Gemini |
| `offline` | Offline extractive baseline (no API) |

With no key, the app runs in offline mode. Restart the server after changing `.env`; the sidebar footer shows the active model.

## 5. Run the tests (no key needed)
```bash
python -m pytest -q
```
Expected: `22 passed`.

## 6. Start the web app
```bash
uvicorn qa_assistant.api:app --reload
```
Open <http://localhost:8000>. The header shows which mode is active.
- Click an example question, or type your own and press **Ask**.
- Expand **Retrieved context** to see which sections were used.
- API docs: <http://localhost:8000/docs>

Stop the server with `Ctrl+C`.

## 7. Use the command line instead
```bash
python -m qa_assistant.cli "Can employees carry forward more than 10 days of leave?"
python -m qa_assistant.cli --show-context "Who approves an expense of ₹1,00,000?"
python -m qa_assistant.cli            # interactive; type 'exit' to quit
python -m qa_assistant.cli --mode offline "How much is the home office stipend?"
python -m qa_assistant.cli --mode gemini "per diem for international trips"
```

## 8. Call the API directly
```bash
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the company dress code?"}'
```

## 9. Run the evaluation
```bash
python -m qa_assistant.evaluate --mode offline   # free baseline
python -m qa_assistant.evaluate --mode gemini    # uses Gemini (48 API calls)
```
Reports are written to `eval/results/results_<mode>.md` and `.json`.

## 10. Use your own knowledge base
Put `.md` or `.txt` files in `knowledge_base/` (use `##` headings to split sections), or point `QA_KB_DIR` at another folder, then restart the server. If you change the company, update the first line of `SYSTEM_PROMPT` in `qa_assistant/prompts.py`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Sidebar says "Offline mode" but you set a key | Check `.env` is in the project root and the variable is `GEMINI_API_KEY`; restart the server. |
| `502 Gemini rejected the credentials` | Key is wrong or revoked. |
| `Gemini model '…' was not found` | Run `python -m qa_assistant.gemini_models` and set `GEMINI_MODEL` to a listed name. |
| `502 rate limit reached` | Wait and retry; lower volume. |
| `Address already in use` | `uvicorn qa_assistant.api:app --port 8001` |
| `ModuleNotFoundError: qa_assistant` | Run commands from the project root with the venv activated. |

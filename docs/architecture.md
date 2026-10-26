# Architecture and Data Flow

Diagrams use [Mermaid](https://mermaid.js.org/); GitHub and VS Code (with a Mermaid extension) render them automatically.

## 1. Architecture diagram

```mermaid
flowchart LR
    subgraph Clients
        UI["Web UI<br/>static/index.html"]
        CLI["CLI<br/>qa_assistant.cli"]
        EVAL["Evaluator<br/>qa_assistant.evaluate"]
    end

    subgraph Server["FastAPI server"]
        API["/api/ask<br/>/api/health<br/>/api/documents"]
    end

    subgraph Core["Core pipeline"]
        PIPE["QAAssistant<br/>pipeline.py"]
        KB["Knowledge-base loader<br/>knowledge_base.py"]
        RET["BM25 retriever<br/>retriever.py"]
        PROMPT["Prompt + JSON schema<br/>prompts.py"]
        GEM["GeminiAnswerer<br/>answerers.py"]
        OFF["ExtractiveAnswerer<br/>(offline fallback)"]
        CFG["Settings<br/>config.py / .env"]
    end

    FILES[("knowledge_base/*.md")]
    GEMINI{{"Gemini API<br/>GEMINI_MODEL"}}
    QS[("eval/questions.json")]
    RES[("eval/results/*.md, *.json")]

    UI -- "HTTP JSON" --> API --> PIPE
    CLI --> PIPE
    EVAL --> PIPE
    QS --> EVAL --> RES
    CFG --> PIPE
    FILES --> KB --> RET
    PIPE --> RET
    PIPE -- "mode = gemini" --> GEM
    PIPE -- "mode = offline" --> OFF
    PROMPT --> GEM
    GEM -- "generate_content<br/>JSON schema" --> GEMINI
```

**Components**

| Component | Responsibility |
|---|---|
| Web UI / CLI / Evaluator | Three front-ends over the same `QAAssistant.ask()` call. |
| FastAPI server | Validates requests, maps errors to HTTP codes, serves the UI. |
| Knowledge-base loader | Splits Markdown files into section chunks with stable ids. |
| BM25 retriever | Ranks chunks for a question; returns top-k with scores. |
| Prompt module | Grounding system prompt, context formatting, answer JSON schema. |
| GeminiAnswerer | One Gemini `generate_content` call with the system prompt, context and JSON schema; handles safety blocks/errors. |
| ExtractiveAnswerer | No-LLM baseline: coverage check + sentence extraction. |
| Pipeline | Orchestrates the steps and verifies citations. |

## 2. Data flow model

### 2.1 Start-up (ingestion)

```mermaid
flowchart LR
    A["knowledge_base/*.md"] --> B["Split on '## ' headings"]
    B --> C["Chunk(id, source, title, section, text)<br/>one per ## section"]
    C --> D["Tokenize + stem<br/>(heading weighted x2)"]
    D --> E["BM25 index<br/>term frequencies, IDF, avg length"]
```

### 2.2 Question answering (per request)

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant UI as Web UI / CLI
    participant P as QAAssistant
    participant R as BM25 Retriever
    participant A as Answerer
    participant C as Gemini API

    U->>UI: Question
    UI->>P: ask(question)
    P->>P: Validate (non-empty, ≤ 1000 chars)
    P->>R: search(question, top_k=5)
    R-->>P: Ranked chunks + scores
    alt No chunk matches
        P-->>UI: status = not_found (no model call)
    else Chunks found
        P->>A: answer(question, chunks)
        alt Gemini mode
            A->>C: system = grounding rules<br/>user = <context>chunks</context><question>
            C-->>A: JSON {status, answer, citations, missing_information}
        else Offline mode
            A->>A: IDF-weighted coverage check<br/>extract best sentences
        end
        A-->>P: Draft
        P->>P: Drop citations not in retrieved set
        P-->>UI: Answer (status, answer, sources, retrieved context)
    end
    UI-->>U: Badge + answer + sources
```

### 2.3 Data at each stage

| Stage | Data | Example |
|---|---|---|
| Input | question string | `"Can employees carry forward more than 10 days of leave?"` |
| Tokens | stemmed terms + synonyms | `employee carry forward 10 day leave` |
| Retrieval | `[(chunk_id, score)]` | `leave_policy#carry-forward-of-annual-leave 17.4`, `leave_policy#sick-leave 10.4`, … |
| Prompt | system rules + `<context>` + `<question>` | see [prompt_design.md](prompt_design.md) |
| Model output | JSON per `ANSWER_SCHEMA` | `{"status":"answered","answer":"No…","citations":["leave_policy#carry-forward-of-annual-leave"],"missing_information":""}` |
| Response | `Answer` JSON | adds sources' text, scores, mode, latency, notes |

### 2.4 Decision logic for "answer not available"

```mermaid
flowchart TD
    Q["Question"] --> R{"Any chunk with<br/>BM25 score > 0?"}
    R -- No --> NF["not_found<br/>(no model call)"]
    R -- Yes --> M{"Mode"}
    M -- gemini --> L{"Gemini: do the excerpts<br/>contain the answer?"}
    L -- Fully --> ANS["answered + citations"]
    L -- Partly --> PAR["partial + missing_information"]
    L -- No --> NF2["not_found + what is missing"]
    L -- "Refusal" --> NF3["not_found"]
    M -- offline --> O{"IDF-weighted<br/>coverage ≥ 0.65?"}
    O -- No --> NF4["not_found"]
    O -- Yes --> EXT["answered (extracted sentences)"]
```

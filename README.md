# ⚡ SOP Execution Agent

> An intelligent, LLM-powered agent that parses Standard Operating Procedure PDFs, embeds them into a vector store, and executes workflows step-by-step with full auditability, human escalation, and RAGAS evaluation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI (app.py)                       │
│  ┌──────────┐  ┌────────────────┐  ┌────────────────────────┐  │
│  │  Upload   │  │  Chat Panel    │  │  Sidebar Tracker       │  │
│  │  PDF      │  │  (user ↔ agent)│  │  ✅⏳🚨 step status   │  │
│  └────┬─────┘  └───────┬────────┘  └────────────────────────┘  │
│       │                │                                        │
├───────┼────────────────┼────────────────────────────────────────┤
│       ▼                ▼                                        │
│  ┌─────────┐    ┌─────────────┐                                 │
│  │ PDF     │    │  LangGraph  │◄──── System Prompt              │
│  │ Loader  │    │  Agent      │      (agent/prompts.py)         │
│  │         │    │  (ReAct)    │                                 │
│  └────┬────┘    └──────┬──────┘                                 │
│       │                │                                        │
│       ▼                ▼                                        │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────┐            │
│  │ Embedder│◄──►│  5 Tools    │───►│ State Tracker│            │
│  │ ChromaDB│    │ retrieve    │    │ completed    │            │
│  │         │    │ execute     │    │ flags        │            │
│  └─────────┘    │ explain     │    │ logs         │            │
│                 │ status      │    └──────────────┘            │
│                 │ escalate    │                                 │
│                 └─────────────┘                                 │
│                                                                 │
│  ┌─────────────────────────────────────────┐                    │
│  │  RAGAS Evaluation (eval/ragas_eval.py)  │                    │
│  │  faithfulness · relevancy · recall      │                    │
│  └─────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd sop-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Edit the `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

---

## How to Run

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## How to Use

1. **Upload** — Drop an SOP PDF into the sidebar uploader. The agent parses sections, embeds them into ChromaDB, and initialises the step tracker.

2. **Chat** — Ask questions, execute steps, or request explanations in the chat panel:
   - `"What are the first 3 steps?"` → retrieves and summarises
   - `"Execute step_1"` → marks step as done, updates tracker
   - `"Why does step_2 exist?"` → LLM-powered explanation
   - `"Check my progress"` → full status report

3. **Track** — Watch the sidebar update in real-time with ✅ completed, ⏳ pending, and 🚨 flagged steps plus a progress bar.

4. **Evaluate** — Click **Run RAGAS Eval** in the sidebar to benchmark retrieval quality with faithfulness, answer relevancy, and context recall metrics.

---

## Tech Stack

| Layer             | Technology                          |
|-------------------|-------------------------------------|
| Frontend          | Streamlit                           |
| LLM               | OpenAI GPT-4o-mini                  |
| Embeddings         | OpenAI text-embedding-3-small       |
| Vector Store       | ChromaDB (persistent)               |
| Agent Framework    | LangGraph (StateGraph + ToolNode)   |
| PDF Parsing        | pdfplumber                          |
| Evaluation         | RAGAS (faithfulness, relevancy, recall) |
| Orchestration      | LangChain                           |

---

## Why This Is Production-Grade

- **Section-aware chunking** — Regex heuristics detect real SOP sections (Step N, numbered headings, ALL CAPS headers) instead of naive fixed-token splitting, preserving semantic boundaries.

- **LangGraph stateful execution** — A proper ReAct state machine with conditional routing ensures the agent always retrieves context before executing and checks status after every action.

- **Human escalation loop** — Ambiguous, risky, or compliance-sensitive steps are automatically flagged for human review before the agent proceeds, with full audit logging.

- **RAGAS evaluation metrics** — Built-in evaluation pipeline measures faithfulness, answer relevancy, and context recall so you can quantify retrieval quality and catch regressions.

---

## Project Structure

```
sop-agent/
├── app.py                  # Streamlit UI entry point
├── ingestion/
│   ├── pdf_loader.py       # Section-aware PDF parser
│   └── embedder.py         # ChromaDB + OpenAI embeddings
├── agent/
│   ├── graph.py            # LangGraph agent state machine
│   ├── tools.py            # 5 LangChain tools
│   └── prompts.py          # System + explain prompts
├── state/
│   └── tracker.py          # Step completion & audit tracker
├── eval/
│   └── ragas_eval.py       # RAGAS evaluation pipeline
├── data/                   # Drop SOP PDFs here
├── .env                    # API key configuration
└── requirements.txt        # Python dependencies
```

---

## License

MIT

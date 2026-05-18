"""
SOP Execution Agent — Streamlit UI

Main entry point: streamlit run app.py
"""

import os
import shutil
import sys

import streamlit as st
from dotenv import load_dotenv

# Load .env before any OpenAI imports
load_dotenv()

# ------------------------------------------------------------------ #
#  Page config (must be first Streamlit call)                          #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="SOP Agent | Windflow Demo",
    page_icon="⚡",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  Custom CSS injection                                                #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Force dark theme on main app container */
.stApp {
    background-color: #0d0f1a;
    color: #e5e7eb;
}

/* Main block container */
.block-container {
    padding-top: 2rem;
}


/* Ensure metric cards are visible */
[data-testid="stMetric"] {
    background-color: #1a1d2e;
    border: 1px solid #2d3147;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #9ca3af !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; }

/* Divider */
hr {
    border-color: #1e2130;
}

/* Caption text */
.stCaption { color: #6b7280 !important; }

/* Sidebar text */
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #d1d5db !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* File uploader — force dark theme on the inner drop-zone */
[data-testid="stFileUploader"] {
    background-color: #1a1d2e !important;
    border-radius: 10px;
}
[data-testid="stFileUploadDropzone"] {
    background-color: #1a1d2e !important;
    border: 1px dashed #4f46e5 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p {
    color: #9ca3af !important;
}
[data-testid="stFileUploadDropzone"] button {
    background-color: #2d3147 !important;
    color: #e5e7eb !important;
    border: 1px solid #4f46e5 !important;
    border-radius: 6px !important;
}

/* Success / error messages */
[data-testid="stAlert"] { border-radius: 8px; }

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0f1117;
    border-right: 1px solid #1e2130;
}

/* Step tracker items */
.step-complete {
    color: #22c55e;
    font-size: 13px;
    padding: 4px 0;
    border-left: 3px solid #22c55e;
    padding-left: 8px;
    margin: 2px 0;
}
.step-pending {
    color: #6b7280;
    font-size: 13px;
    padding: 4px 0;
    border-left: 3px solid #374151;
    padding-left: 8px;
    margin: 2px 0;
}
.step-flagged {
    color: #ef4444;
    font-size: 13px;
    padding: 4px 0;
    border-left: 3px solid #ef4444;
    padding-left: 8px;
    margin: 2px 0;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #1a1d2e;
    border: 1px solid #1e2130;
    border-radius: 12px;
    margin-bottom: 8px;
    padding: 4px;
}

/* Chat input */
[data-testid="stChatInput"] {
    border: 1px solid #2d3147;
    border-radius: 12px;
    background-color: #1a1d2e;
}

/* Quick prompt buttons */
.stButton > button {
    background-color: #1a1d2e;
    color: #9ca3af;
    border: 1px solid #2d3147;
    border-radius: 8px;
    font-size: 12px;
    padding: 6px 12px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background-color: #2d3147;
    color: #ffffff;
    border-color: #4f46e5;
}

/* Progress bar */
[data-testid="stProgressBar"] > div > div {
    background-color: #4f46e5;
}

/* RAGAS table */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2130;
    border-radius: 8px;
}

/* Section headers in sidebar */
.sidebar-section-header {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4b5563;
    margin: 16px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  API key guard                                                       #
# ------------------------------------------------------------------ #
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key or api_key == "your_openai_api_key_here":
    st.error(
        "🔑 OpenAI API key not found. Add it to your `.env` file:\n\n"
        "```\nOPENAI_API_KEY=sk-...\n```"
    )
    st.stop()

# ------------------------------------------------------------------ #
#  Imports (after env is loaded)                                       #
# ------------------------------------------------------------------ #
from ingestion.pdf_loader import SOPLoader
from ingestion.embedder import SOPEmbedder
from agent.graph import create_agent_graph, run_agent
from state.tracker import SOPStateTracker
from eval.ragas_eval import run_evaluation, format_eval_results

# ------------------------------------------------------------------ #
#  Session state initialisation                                        #
# ------------------------------------------------------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tracker" not in st.session_state:
    st.session_state.tracker = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "sop_loaded" not in st.session_state:
    st.session_state.sop_loaded = False
if "step_list" not in st.session_state:
    st.session_state.step_list = []
if "sop_filename" not in st.session_state:
    st.session_state.sop_filename = ""


# ------------------------------------------------------------------ #
#  Reset project helper                                                #
# ------------------------------------------------------------------ #
def reset_project():
    """Clear all session state and delete the chroma_db directory."""
    st.session_state.sop_loaded = False
    st.session_state.messages = []
    st.session_state.tracker = None
    st.session_state.embedder = None
    st.session_state.graph = None
    st.session_state.step_list = []
    st.session_state.sop_filename = ""
    shutil.rmtree("./chroma_db", ignore_errors=True)
    st.rerun()


# ------------------------------------------------------------------ #
#  Header                                                              #
# ------------------------------------------------------------------ #
col_title, col_action = st.columns([5, 1])

with col_title:
    st.markdown("""
    <div style='padding: 8px 0 16px 0;'>
        <span style='font-size:28px; font-weight:700;
        color:#ffffff;'>⚡ SOP Execution Agent</span><br>
        <span style='font-size:13px; color:#6b7280;'>
        Powered by LangGraph + RAG — Windflow.ai Demo
        </span>
    </div>
    """, unsafe_allow_html=True)

with col_action:
    if st.session_state.sop_loaded:
        if st.button("⟳ New Project", key="new_project_btn",
                     help="Close current SOP and start fresh"):
            reset_project()

st.divider()

# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    # ── Sidebar brand header ──────────────────────────────────────
    st.markdown("""
    <div style='padding:12px 0 4px 0;
    font-size:13px; font-weight:600;
    color:#4f46e5; letter-spacing:0.05em;'>
    WINDFLOW.AI
    </div>""", unsafe_allow_html=True)

    # ── Current Project status bar ────────────────────────────────
    if st.session_state.sop_loaded:
        sop_name = st.session_state.sop_filename
        display_name = sop_name[:20] + "..." if len(sop_name) > 20 else sop_name
        proj_col, close_col = st.columns([4, 1])
        with proj_col:
            st.markdown(
                f"<div style='background:#1a1d2e; border:1px solid #2d3147; "
                f"border-radius:8px; padding:8px 12px; font-size:13px; "
                f"color:#d1d5db;'>📂 {display_name}</div>",
                unsafe_allow_html=True,
            )
        with close_col:
            if st.button("✕", key="close_project_btn",
                         help="Close current SOP"):
                reset_project()

    # ── Upload SOP (only when no SOP is loaded) ───────────────────
    if not st.session_state.sop_loaded:
        st.header("📄 Upload SOP")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", type=["pdf"], key="pdf_uploader"
        )

        if uploaded_file is not None:
            # Save to disk
            save_path = os.path.join("data", "uploaded_sop.pdf")
            os.makedirs("data", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Parsing and embedding SOP..."):
                try:
                    # a) Parse PDF
                    loader = SOPLoader(save_path)
                    chunks = loader.load_and_chunk()

                    if not chunks:
                        st.error("❌ No sections found in the PDF. Is it a valid SOP document?")
                        st.stop()

                    # b) Embed chunks
                    embedder = SOPEmbedder()
                    num_stored = embedder.ingest(chunks)

                    # c) Extract step list
                    step_list = loader.extract_step_list()

                    # d) Initialise tracker
                    tracker = SOPStateTracker(step_list)

                    # e) Build agent graph
                    graph = create_agent_graph(embedder, tracker)

                    # f) Store in session state
                    st.session_state.embedder = embedder
                    st.session_state.tracker = tracker
                    st.session_state.graph = graph
                    st.session_state.sop_loaded = True
                    st.session_state.step_list = step_list
                    st.session_state.messages = []
                    st.session_state.sop_filename = uploaded_file.name

                    st.success(f"✅ SOP loaded! {num_stored} sections indexed.")
                except Exception as exc:
                    st.error(f"❌ Failed to load SOP: {exc}")

    # ── Step Tracker ───────────────────────────────────────────────
    if st.session_state.sop_loaded and st.session_state.tracker:
        st.divider()
        st.header("📊 Step Tracker")

        tracker = st.session_state.tracker
        status = tracker.get_status()

        completed_ids = set(tracker.completed)
        flagged_ids = {f["step_id"] for f in tracker.flags}

        for step in st.session_state.step_list:
            sid = step["id"]
            title = step["title"]
            # Truncate long titles
            display_title = title[:28] + "..." if len(title) > 28 else title

            if sid in completed_ids:
                st.markdown(
                    f"<div class='step-complete'>✓ {display_title}</div>",
                    unsafe_allow_html=True,
                )
            elif sid in flagged_ids:
                st.markdown(
                    f"<div class='step-flagged'>⚠ {display_title}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='step-pending'>○ {display_title}</div>",
                    unsafe_allow_html=True,
                )

        # Progress bar
        total = status["total"]
        done = status["completed_count"]
        progress = done / total if total > 0 else 0.0
        st.progress(progress)
        st.caption(f"Steps Complete: {done} / {total}")

    # ── RAGAS Evaluation ──────────────────────────────────────────
    if st.session_state.sop_loaded and st.session_state.embedder:
        st.divider()
        st.header("🧪 Run Evaluation")

        if st.button("Run RAGAS Eval", key="ragas_btn"):
            with st.spinner("Running RAGAS evaluation... this may take a minute."):
                try:
                    results_df = run_evaluation(st.session_state.embedder)
                    st.dataframe(results_df, use_container_width=True)

                    # Metric cards
                    metric_columns = {
                        "faithfulness": "Faithfulness",
                        "answer_relevancy": "Answer Relevancy",
                        "context_recall": "Context Recall",
                    }

                    metric_values = {}
                    for col, label in metric_columns.items():
                        if col in results_df.columns:
                            mean_val = results_df[col].mean()
                            metric_values[label] = mean_val
                        else:
                            metric_values[label] = None

                    cols = st.columns(3)
                    metric_items = list(metric_values.items())
                    for i, (label, val) in enumerate(metric_items):
                        with cols[i]:
                            if val is not None and not (isinstance(val, float) and str(val) == "nan"):
                                import math
                                if math.isnan(val):
                                    st.metric(label, "—")
                                else:
                                    st.metric(label, f"{val:.2f}")
                            else:
                                st.metric(label, "—")

                    st.text(format_eval_results(results_df))
                except Exception as exc:
                    st.error(f"⚠️ Evaluation failed: {exc}")

# ------------------------------------------------------------------ #
#  Main panel — chat interface                                         #
# ------------------------------------------------------------------ #
if not st.session_state.sop_loaded:
    st.markdown("""
    <div style='text-align:center; padding: 80px 40px;'>
        <div style='font-size:48px; margin-bottom:16px;'>⚡</div>
        <div style='font-size:20px; font-weight:600;
        color:#ffffff; margin-bottom:8px;'>
        Ready to automate your workflow
        </div>
        <div style='font-size:14px; color:#6b7280;
        max-width:400px; margin:0 auto;'>
        Upload a Standard Operating Procedure PDF in the
        sidebar. The agent will parse every section, build
        a knowledge base, and execute your workflow
        step by step.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display chat history — filter out empty/tool-call messages
    for msg in st.session_state.messages:
        # Extract content safely
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if not content or not str(content).strip():
            continue  # skip tool-call intermediate messages

        # Determine role from message type
        role = "assistant"
        if hasattr(msg, "type"):
            if msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            else:
                continue  # skip system/tool messages in display
        elif hasattr(msg, "__class__"):
            class_name = msg.__class__.__name__
            if class_name == "HumanMessage":
                role = "user"
            elif class_name == "AIMessage":
                role = "assistant"
            else:
                continue

        with st.chat_message(role):
            st.markdown(str(content))

    # Chat input
    user_input = st.chat_input(
        "Ask about a step, execute it, or check progress..."
    )

    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Agent thinking..."):
                try:
                    response_text, updated_messages = run_agent(
                        st.session_state.graph,
                        user_input,
                        st.session_state.messages,
                    )
                    st.markdown(response_text)
                    st.session_state.messages = updated_messages
                except Exception as exc:
                    st.error(f"⚠️ Agent error: {exc}")

        # Rerun to refresh tracker sidebar
        st.rerun()

    # ── Suggested starter prompts ──────────────────────────────────
    st.divider()
    st.markdown("""
    <div style='font-size:11px; color:#4b5563;
    font-weight:600; text-transform:uppercase;
    letter-spacing:0.08em; margin-bottom:8px;'>
    Suggested Actions
    </div>""", unsafe_allow_html=True)

    cols = st.columns(4)

    # Build dynamic prompts from the real step titles
    _steps = st.session_state.step_list
    _s1_id    = _steps[0]["id"]    if len(_steps) > 0 else "step_1"
    _s1_title = _steps[0]["title"] if len(_steps) > 0 else "Step 1"
    _s2_id    = _steps[1]["id"]    if len(_steps) > 1 else "step_2"
    _s2_title = _steps[1]["title"] if len(_steps) > 1 else "Step 2"

    _prompts = [
        "What are the first 3 steps?",
        f"Execute {_s1_id}",
        f"Why does {_s2_title} exist?",
        "Check my progress",
    ]
    _labels = [
        "📋 List first 3 steps",
        f"▶ Execute {_s1_title[:18]}{'...' if len(_s1_title) > 18 else ''}",
        f"❓ Why {_s2_title[:18]}{'...' if len(_s2_title) > 18 else ''}?",
        "📊 Check my progress",
    ]

    for col, (_label, _prompt_text) in zip(cols, zip(_labels, _prompts)):
        with col:
            if st.button(_label, key=f"btn_{_prompt_text}"):
                # Inject the prompt as if the user typed it
                with st.chat_message("user"):
                    st.markdown(_prompt_text)
                with st.chat_message("assistant"):
                    with st.spinner("Agent thinking..."):
                        try:
                            response_text, updated_messages = run_agent(
                                st.session_state.graph,
                                _prompt_text,
                                st.session_state.messages,
                            )
                            st.markdown(response_text)
                            st.session_state.messages = updated_messages
                        except Exception as exc:
                            st.error(f"⚠️ Agent error: {exc}")
                st.rerun()


"""
RAGAS Evaluation — evaluates the RAG pipeline quality using
hardcoded QA pairs and the ragas library.
"""

import logging

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Hardcoded QA pairs (generic — work for any SOP)                     #
# ------------------------------------------------------------------ #

_QA_PAIRS = [
    {
        "question": "What is the first step in the SOP?",
        "ground_truth": "The first step involves initial preparation and setup required before the main procedure begins.",
    },
    {
        "question": "What should be done before starting the process?",
        "ground_truth": "Prerequisites including gathering required documents, verifying access permissions, and reviewing the procedure should be completed before starting.",
    },
    {
        "question": "Who is responsible for executing this procedure?",
        "ground_truth": "The designated process owner or assigned team member is responsible for executing the procedure according to the SOP.",
    },
    {
        "question": "What documents are required for this SOP?",
        "ground_truth": "Required documents typically include authorization forms, checklists, compliance certificates, and any referenced templates.",
    },
    {
        "question": "What are the compliance requirements in this SOP?",
        "ground_truth": "Compliance requirements include adherence to regulatory standards, internal audit procedures, and documentation of all actions taken.",
    },
    {
        "question": "What happens if a step fails?",
        "ground_truth": "If a step fails, the procedure should be halted, the failure documented, and the issue escalated to the appropriate authority for resolution.",
    },
    {
        "question": "What is the final step of the procedure?",
        "ground_truth": "The final step involves verification of all completed actions, sign-off by the responsible authority, and archival of documentation.",
    },
    {
        "question": "Are there any escalation procedures?",
        "ground_truth": "Escalation procedures exist for handling exceptions, compliance concerns, and situations requiring management approval.",
    },
]


# ------------------------------------------------------------------ #
#  Evaluation runner                                                   #
# ------------------------------------------------------------------ #

def run_evaluation(embedder) -> pd.DataFrame:
    """Run RAGAS evaluation against the ingested SOP chunks.

    Args:
        embedder: A SOPEmbedder instance with chunks already ingested.

    Returns:
        A pandas DataFrame with evaluation metric scores.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for qa in _QA_PAIRS:
        question = qa["question"]
        ground_truth = qa["ground_truth"]

        # Retrieve context from the embedder
        try:
            search_results = embedder.search(question, k=3)
        except Exception as exc:
            logger.error("Search failed for '%s': %s", question, exc)
            search_results = []

        contexts = [r["content"] for r in search_results] if search_results else ["No context found."]
        joined_context = "\n\n".join(contexts)

        # Generate answer using LLM
        prompt = (
            f"Answer this question based only on the context.\n"
            f"Context: {joined_context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        try:
            response = llm.invoke(prompt)
            answer = response.content
        except Exception as exc:
            logger.error("⚠️ LLM Error for question '%s': %s", question, exc)
            answer = f"⚠️ LLM Error: {exc}. Please retry."

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)

    # Build the ragas Dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    # Run evaluation
    try:
        results = evaluate(
            dataset=eval_dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
        )
        results_df = results.to_pandas()
    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        # Return a fallback dataframe with the raw data
        results_df = pd.DataFrame({
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
            "note": [f"Evaluation error: {exc}"] * len(questions),
        })

    return results_df


# ------------------------------------------------------------------ #
#  Results formatter                                                   #
# ------------------------------------------------------------------ #

def format_eval_results(df: pd.DataFrame) -> str:
    """Format evaluation results into a human-readable summary.

    Args:
        df: DataFrame returned by run_evaluation().

    Returns:
        Formatted summary string.
    """
    summary_lines = ["📊 RAGAS Evaluation Summary", "=" * 40]

    # Check which metric columns exist in the dataframe
    metric_columns = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_recall": "Context Recall",
    }

    for col, label in metric_columns.items():
        if col in df.columns:
            mean_score = df[col].mean()
            summary_lines.append(f"  {label}: {mean_score:.4f}")
        else:
            summary_lines.append(f"  {label}: N/A (not computed)")

    summary_lines.append("=" * 40)
    summary_lines.append(f"  Total Questions Evaluated: {len(df)}")

    return "\n".join(summary_lines)

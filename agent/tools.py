"""
SOP Agent Tools — five LangChain tools wired to the embedder and state tracker.

Call setup_tools(embedder, tracker) to get a configured list of tools
ready to be bound to the LangGraph agent.
"""

import logging

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .prompts import EXPLAIN_PROMPT

logger = logging.getLogger(__name__)


def setup_tools(embedder, tracker) -> list:
    """Create and return the five SOP agent tools.

    Args:
        embedder: A SOPEmbedder instance (provides search / get_all_chunks).
        tracker:  A SOPStateTracker instance (provides mark_complete / flag / status).

    Returns:
        List of five LangChain tool callables.
    """

    # ------------------------------------------------------------------ #
    #  Tool 1: retrieve_step                                               #
    # ------------------------------------------------------------------ #
    @tool
    def retrieve_step(query: str) -> str:
        """Retrieve the most relevant SOP steps and context for a given task or question. Always call this before executing any step."""
        try:
            results = embedder.search(query, k=3)
        except Exception as exc:
            return f"⚠️ LLM Error: {exc}. Please retry."

        if not results:
            return "No relevant SOP sections found for that query."

        formatted = "RETRIEVED CONTEXT:\n\n"
        for r in results:
            title = r.get("metadata", {}).get("section_title", "Unknown")
            content = r.get("content", "")
            formatted += f"[Section: {title}]\n{content}\n\n"
        return formatted.strip()

    # ------------------------------------------------------------------ #
    #  Tool 2: execute_step                                                #
    # ------------------------------------------------------------------ #
    @tool
    def execute_step(step_id: str) -> str:
        """Mark a specific SOP step as executed and completed. Use the step_id exactly as returned by retrieve_step (e.g. 'step_1', 'step_3')."""
        try:
            # Look up the title for a friendlier confirmation
            id_to_title = {s["id"]: s["title"] for s in tracker.steps}
            title = id_to_title.get(step_id, step_id)

            result = tracker.mark_complete(step_id)
            next_pending = tracker.get_next_pending()

            if next_pending:
                next_label = f"{next_pending['id']} — {next_pending['title']}"
            else:
                next_label = "ALL STEPS COMPLETE"

            return f"✅ Executed: {step_id} — {title}. Next pending: {next_label}"
        except Exception as exc:
            return f"⚠️ Error executing step: {exc}. Please retry."

    # ------------------------------------------------------------------ #
    #  Tool 3: explain_reasoning                                           #
    # ------------------------------------------------------------------ #
    @tool
    def explain_reasoning(step_id: str) -> str:
        """Explain why a specific SOP step exists, what it requires, and what happens if skipped. Use this when the user asks 'why' or wants to understand a step."""
        try:
            all_chunks = embedder.get_all_chunks()
        except Exception as exc:
            return f"⚠️ LLM Error: {exc}. Please retry."

        # Find the matching chunk
        target_chunk = None
        for chunk in all_chunks:
            if chunk["id"] == step_id:
                target_chunk = chunk
                break

        if target_chunk is None:
            return "Step not found in SOP."

        prompt_text = EXPLAIN_PROMPT.format(content=target_chunk["content"])

        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(prompt_text)
            return response.content
        except Exception as exc:
            return f"⚠️ LLM Error: {exc}. Please retry."

    # ------------------------------------------------------------------ #
    #  Tool 4: check_status                                                #
    # ------------------------------------------------------------------ #
    @tool
    def check_status(_: str = "") -> str:
        """Check the current execution status of the SOP. Returns total steps, completed steps, and any flags."""
        try:
            status = tracker.get_status()
            pending = status["pending"]    # list of {id, title}
            completed = status["completed"]  # list of {id, title}
            flags = status["flags"]

            pending_lines = "\n".join(
                f"    • {s['id']} — {s['title']}" for s in pending
            ) or "    None"

            completed_lines = "\n".join(
                f"    ✓ {s['id']} — {s['title']}" for s in completed
            ) or "    None"

            flagged_lines = "\n".join(
                f"    ⚠ {f['step_id']} — {f['reason']}" for f in flags
            ) or "    None"

            report = (
                f"📊 SOP Progress Report:\n"
                f"  Total Steps   : {status['total']}\n"
                f"  Completed     : {status['completed_count']}\n"
                f"  Pending       : {len(pending)}\n"
                f"  Flagged       : {len(flags)}\n\n"
                f"Completed steps:\n{completed_lines}\n\n"
                f"Pending steps:\n{pending_lines}\n\n"
                f"Flagged steps:\n{flagged_lines}"
            )
            return report
        except Exception as exc:
            return f"⚠️ Error checking status: {exc}. Please retry."

    # ------------------------------------------------------------------ #
    #  Tool 5: escalate_to_human                                           #
    # ------------------------------------------------------------------ #
    @tool
    def escalate_to_human(step_id_and_reason: str) -> str:
        """Flag an ambiguous, risky, or compliance-sensitive SOP step for human review before proceeding. Input format: 'step_id|reason for escalation'."""
        try:
            parts = step_id_and_reason.split("|", 1)
            if len(parts) != 2:
                return "⚠️ Invalid format. Use: 'step_id|reason for escalation'"

            step_id = parts[0].strip()
            reason = parts[1].strip()

            tracker.flag_for_human(step_id, reason)
            return (
                f"🚨 ESCALATED: Step {step_id} flagged for human review. "
                f"Reason: {reason}. Pausing execution until resolved."
            )
        except Exception as exc:
            return f"⚠️ Error escalating: {exc}. Please retry."

    return [retrieve_step, execute_step, explain_reasoning, check_status, escalate_to_human]

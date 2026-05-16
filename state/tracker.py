"""
SOP State Tracker — tracks step completion, flags, and audit logs.

Uses only the Python standard library (no external dependencies).
"""

import json
from datetime import datetime


class SOPStateTracker:
    """Tracks the execution state of an SOP workflow.

    Maintains a list of steps, which ones are completed, any flags
    raised for human review, and a full audit log of every action.
    """

    def __init__(self, steps: list[dict]):
        """Initialise the tracker with a list of step dicts.

        Args:
            steps: List of dicts, each with keys 'id', 'title', 'description'.
                   Example: [{"id": "step_1", "title": "...", "description": "..."}]
        """
        self.steps: list[dict] = steps
        self.completed: list[str] = []
        self.flags: list[dict] = []
        self.logs: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Public methods                                                      #
    # ------------------------------------------------------------------ #

    def mark_complete(self, step_id: str) -> str:
        """Mark a step as completed.

        Args:
            step_id: The id of the step to mark (e.g. 'step_1').

        Returns:
            Confirmation string.
        """
        if step_id not in self.completed:
            self.completed.append(step_id)
        self.log(step_id, "mark_complete", detail=f"Step {step_id} marked as completed")
        return f"Step {step_id} marked complete."

    def flag_for_human(self, step_id: str, reason: str) -> str:
        """Flag a step for human review.

        Args:
            step_id: The id of the step to flag.
            reason:  Why the step needs human attention.

        Returns:
            Confirmation string with the reason.
        """
        flag_entry = {
            "step_id": step_id,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        self.flags.append(flag_entry)
        self.log(step_id, "flag_for_human", detail=reason)
        return f"Step {step_id} flagged: {reason}"

    def get_status(self) -> dict:
        """Return a summary of the current SOP execution status.

        Returns:
            Dict with keys: total, completed_count, pending (list of ids),
            flags (list of flag dicts).
        """
        all_ids = [s["id"] for s in self.steps]
        pending = [sid for sid in all_ids if sid not in self.completed]
        return {
            "total": len(self.steps),
            "completed_count": len(self.completed),
            "pending": pending,
            "flags": self.flags,
        }

    def get_next_pending(self) -> str | None:
        """Return the id of the first step not yet completed.

        Returns:
            The step id string, or None if all steps are done.
        """
        for step in self.steps:
            if step["id"] not in self.completed:
                return step["id"]
        return None

    def log(self, step_id: str, action: str, detail: str = "") -> None:
        """Append an entry to the internal audit log.

        Args:
            step_id: Which step the action relates to.
            action:  Action name (e.g. 'mark_complete', 'flag_for_human').
            detail:  Optional human-readable detail string.
        """
        entry = {
            "step_id": step_id,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "detail": detail,
        }
        self.logs.append(entry)

    def to_json(self) -> str:
        """Serialise the full tracker state to a JSON string.

        Returns:
            Pretty-printed JSON string with indent=2.
        """
        state = {
            "steps": self.steps,
            "completed": self.completed,
            "flags": self.flags,
            "logs": self.logs,
        }
        return json.dumps(state, indent=2)

"""
rab/apps/compliance_workbench.py

ARE App: Compliance Workbench

Exposes a minimal compliance action surface to the agent under evaluation.
The agent calls these methods to respond to a regulatory change scenario.

Registered actions:
  acknowledge_change        — agent signals it has read and understood the update
  classify_required_action  — agent declares what type of action is needed
  submit_response           — agent submits its full compliance response
  escalate                  — agent escalates to legal/compliance team
  request_clarification     — agent flags ambiguity in the regulatory update

State is maintained per scenario run and validated against ground_truth
by rab/scorer.py at scenario completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Action type registry
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = {
    "policy_change",
    "process_change",
    "reporting_change",
    "tech_data_change",
    "training_change",
    "escalate",
    "acknowledge",
    "other",
}


# ---------------------------------------------------------------------------
# Workbench state
# ---------------------------------------------------------------------------

@dataclass
class WorkbenchState:
    """Mutable state accumulated during a single scenario run."""
    scenario_id:       str
    acknowledged:      bool                  = False
    classified_types:  list[str]             = field(default_factory=list)
    submitted_facts:   list[str]             = field(default_factory=list)
    escalated:         bool                  = False
    clarifications:    list[str]             = field(default_factory=list)
    full_response:     dict[str, Any] | None = None
    action_log:        list[dict]            = field(default_factory=list)
    completed:         bool                  = False

    def log(self, action: str, payload: dict) -> None:
        self.action_log.append({
            "action":    action,
            "payload":   payload,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })


# ---------------------------------------------------------------------------
# App class — registered with ARE
# ---------------------------------------------------------------------------

class ComplianceWorkbench:
    """ARE App: simulates a compliance team's response workbench.

    In an ARE scenario, this app is instantiated once per run.
    The agent interacts with it via the methods below.
    """

    APP_NAME = "compliance_workbench"

    def __init__(self, scenario_id: str) -> None:
        self.state = WorkbenchState(scenario_id=scenario_id)

    # ------------------------------------------------------------------
    # Actions available to the agent
    # ------------------------------------------------------------------

    def acknowledge_change(self, summary: str = "") -> dict:
        """Agent acknowledges it has read and understood the regulatory update.

        Args:
            summary: Optional brief acknowledgement text from the agent.

        Returns:
            Confirmation dict.
        """
        self.state.acknowledged = True
        self.state.log("acknowledge_change", {"summary": summary})
        return {
            "status":  "acknowledged",
            "message": "Regulatory change acknowledged. Proceed with classification.",
        }

    def classify_required_action(
        self,
        action_types: list[str],
        rationale:    str = "",
    ) -> dict:
        """Agent declares what type(s) of compliance action are required.

        Args:
            action_types: List of action type strings from VALID_ACTION_TYPES.
            rationale:    Agent's reasoning for this classification.

        Returns:
            Confirmation with any unrecognised types flagged.
        """
        invalid = [t for t in action_types if t not in VALID_ACTION_TYPES]
        valid   = [t for t in action_types if t in VALID_ACTION_TYPES]

        self.state.classified_types.extend(valid)
        self.state.log("classify_required_action", {
            "action_types": action_types,
            "rationale":    rationale,
        })

        result: dict[str, Any] = {
            "status":          "classified",
            "accepted_types":  valid,
        }
        if invalid:
            result["warning"] = f"Unrecognised action types ignored: {invalid}"
            result["valid_types"] = sorted(VALID_ACTION_TYPES)

        return result

    def submit_response(
        self,
        action_types:    list[str],
        identified_facts: list[str],
        key_deadline:    str | None = None,
        penalty_summary: str | None = None,
        notes:           str        = "",
    ) -> dict:
        """Agent submits its full compliance response.

        This is the primary scored action. The scorer checks:
          - action_types     against ground_truth.correct_action_types
          - identified_facts against validation.must_identify
          - key_deadline     against ground_truth.key_deadline

        Args:
            action_types:     Final list of action types the agent commits to.
            identified_facts: Key facts the agent extracted from the update.
            key_deadline:     The primary compliance deadline identified (YYYY-MM-DD).
            penalty_summary:  Agent's summary of non-compliance consequences.
            notes:            Any additional agent notes.

        Returns:
            Submission receipt.
        """
        self.state.submitted_facts  = identified_facts
        self.state.full_response = {
            "action_types":     action_types,
            "identified_facts": identified_facts,
            "key_deadline":     key_deadline,
            "penalty_summary":  penalty_summary,
            "notes":            notes,
        }
        self.state.classified_types.extend(
            t for t in action_types if t not in self.state.classified_types
        )
        self.state.completed = True
        self.state.log("submit_response", self.state.full_response)

        return {
            "status":  "submitted",
            "message": "Response recorded. Scenario will now be evaluated.",
            "receipt": {
                "scenario_id":     self.state.scenario_id,
                "action_types":    action_types,
                "facts_submitted": len(identified_facts),
                "deadline":        key_deadline,
            },
        }

    def escalate(self, reason: str, urgency: str = "normal") -> dict:
        """Agent escalates the matter to legal / compliance leadership.

        Args:
            reason:  Why the agent is escalating.
            urgency: 'normal' | 'urgent' | 'critical'

        Returns:
            Escalation confirmation.
        """
        self.state.escalated = True
        self.state.classified_types.append("escalate")
        self.state.log("escalate", {"reason": reason, "urgency": urgency})
        return {
            "status":  "escalated",
            "urgency": urgency,
            "message": f"Matter escalated. Reason logged: {reason[:120]}",
        }

    def request_clarification(self, question: str) -> dict:
        """Agent flags ambiguity and requests clarification.

        Args:
            question: What the agent needs clarified.

        Returns:
            Acknowledgement (no further information provided — agent must proceed).
        """
        self.state.clarifications.append(question)
        self.state.log("request_clarification", {"question": question})
        return {
            "status":  "noted",
            "message": (
                "Clarification request logged. No additional information is available "
                "for this scenario. Proceed with best available information."
            ),
        }

    # ------------------------------------------------------------------
    # State inspection (used by scorer, not by agent)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return the current workbench state as a plain dict."""
        return {
            "scenario_id":      self.state.scenario_id,
            "acknowledged":     self.state.acknowledged,
            "classified_types": list(set(self.state.classified_types)),
            "submitted_facts":  self.state.submitted_facts,
            "escalated":        self.state.escalated,
            "clarifications":   self.state.clarifications,
            "full_response":    self.state.full_response,
            "action_log":       self.state.action_log,
            "completed":        self.state.completed,
        }

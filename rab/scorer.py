"""
rab/scorer.py

Scores an agent's ComplianceWorkbench state against a scenario's
ground_truth and validation spec.

Scoring dimensions:
  action_coverage   — fraction of required action types correctly identified
  fact_extraction   — fraction of must_identify facts surfaced
  deadline_accuracy — bool: correct key_deadline identified
  critical_miss     — count of must_not_miss items omitted

Outcome:
  PASS            action_coverage == 1.0 and critical_miss == 0
  PARTIAL         action_coverage >= 0.5 and critical_miss <= 1
  FAIL            otherwise
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreResult:
    scenario_id:       str
    outcome:           str          # PASS | PARTIAL | FAIL
    action_coverage:   float        # 0.0 – 1.0
    fact_extraction:   float        # 0.0 – 1.0
    deadline_accurate: bool
    critical_misses:   list[str]
    matched_actions:   list[str]
    missing_actions:   list[str]
    matched_facts:     list[str]
    missing_facts:     list[str]
    notes:             list[str]


def _normalise(s: str) -> str:
    return s.lower().strip()


def _fact_matched(fact: str, submitted: list[str]) -> bool:
    """Fuzzy match: any submitted fact contains key tokens from `fact`."""
    tokens = [t for t in _normalise(fact).split() if len(t) > 3]
    if not tokens:
        return False
    norm_submitted = [_normalise(f) for f in submitted]
    return any(
        all(tok in sub for tok in tokens)
        for sub in norm_submitted
    )


def score_run(
    scenario:       dict,
    workbench_state: dict,
) -> ScoreResult:
    """Score a completed scenario run.

    Args:
        scenario:        The full scenario dict (from are_scenarios.json).
        workbench_state: Output of ComplianceWorkbench.get_state().

    Returns:
        ScoreResult with all scoring dimensions populated.
    """
    scenario_id  = scenario.get("scenario_id", "unknown")
    ground_truth = scenario.get("ground_truth", {})
    validation   = scenario.get("validation", {})
    notes        = []

    # --- Agent outputs ---------------------------------------------------
    agent_types  = set(workbench_state.get("classified_types", []))
    agent_facts  = workbench_state.get("submitted_facts", [])
    agent_deadline = None
    if workbench_state.get("full_response"):
        agent_deadline = workbench_state["full_response"].get("key_deadline")

    # --- Action coverage -------------------------------------------------
    required_types = set(ground_truth.get("correct_action_types", []))
    if not required_types:
        # Fall back to must_act_on from validation
        required_types = set(validation.get("must_act_on", []))

    if required_types:
        matched_actions = list(required_types & agent_types)
        missing_actions = list(required_types - agent_types)
        action_coverage = len(matched_actions) / len(required_types)
    else:
        matched_actions = []
        missing_actions = []
        action_coverage = 1.0  # nothing required → vacuously covered
        notes.append("no required action types defined in ground_truth")

    # --- Fact extraction -------------------------------------------------
    must_identify = validation.get("must_identify", [])
    if must_identify:
        matched_facts = [f for f in must_identify if _fact_matched(f, agent_facts)]
        missing_facts = [f for f in must_identify if not _fact_matched(f, agent_facts)]
        fact_extraction = len(matched_facts) / len(must_identify)
    else:
        matched_facts   = []
        missing_facts   = []
        fact_extraction = 1.0
        notes.append("no must_identify facts defined")

    # --- Deadline accuracy -----------------------------------------------
    expected_deadline = ground_truth.get("key_deadline")
    if expected_deadline and agent_deadline:
        deadline_accurate = _normalise(agent_deadline) == _normalise(expected_deadline)
    elif not expected_deadline:
        deadline_accurate = True   # nothing to check
        notes.append("no key_deadline in ground_truth")
    else:
        deadline_accurate = False  # expected but agent didn't provide
        notes.append(f"agent did not identify key_deadline: {expected_deadline}")

    # --- Critical misses -------------------------------------------------
    must_not_miss    = validation.get("must_not_miss", [])
    critical_misses  = []
    all_agent_text   = " ".join(agent_facts + list(agent_types)).lower()
    if workbench_state.get("full_response"):
        resp = workbench_state["full_response"]
        all_agent_text += " " + " ".join(filter(None, [
            resp.get("penalty_summary", ""),
            resp.get("notes", ""),
            resp.get("key_deadline", ""),
        ])).lower()

    for item in must_not_miss:
        tokens = [t for t in _normalise(item).split() if len(t) > 3]
        if tokens and not any(tok in all_agent_text for tok in tokens):
            critical_misses.append(item)

    # --- Outcome ---------------------------------------------------------
    if action_coverage >= 1.0 and len(critical_misses) == 0:
        outcome = "PASS"
    elif action_coverage >= 0.5 and len(critical_misses) <= 1:
        outcome = "PARTIAL"
    else:
        outcome = "FAIL"

    # Escalation bonus: if agent escalated on a hard scenario, don't penalise
    if (workbench_state.get("escalated")
            and scenario.get("difficulty") == "hard"
            and outcome == "PARTIAL"):
        notes.append("escalation on hard scenario — outcome retained as PARTIAL")

    return ScoreResult(
        scenario_id       = scenario_id,
        outcome           = outcome,
        action_coverage   = round(action_coverage, 3),
        fact_extraction   = round(fact_extraction, 3),
        deadline_accurate = deadline_accurate,
        critical_misses   = critical_misses,
        matched_actions   = matched_actions,
        missing_actions   = missing_actions,
        matched_facts     = matched_facts,
        missing_facts     = missing_facts,
        notes             = notes,
    )


def aggregate_results(score_results: list[ScoreResult]) -> dict:
    """Compute aggregate statistics across all scored scenarios."""
    if not score_results:
        return {}

    n = len(score_results)
    outcomes = [r.outcome for r in score_results]

    by_difficulty: dict[str, list] = {}

    return {
        "total":               n,
        "pass":                outcomes.count("PASS"),
        "partial":             outcomes.count("PARTIAL"),
        "fail":                outcomes.count("FAIL"),
        "pass_rate":           round(outcomes.count("PASS") / n, 3),
        "pass_or_partial_rate":round((outcomes.count("PASS") + outcomes.count("PARTIAL")) / n, 3),
        "avg_action_coverage": round(sum(r.action_coverage for r in score_results) / n, 3),
        "avg_fact_extraction": round(sum(r.fact_extraction for r in score_results) / n, 3),
        "deadline_accuracy":   round(sum(r.deadline_accurate for r in score_results) / n, 3),
        "avg_critical_misses": round(sum(len(r.critical_misses) for r in score_results) / n, 3),
    }

"""tests/test_scorer.py — basic scorer unit tests."""

import pytest
from rab.scorer import score_run, aggregate_results


def _make_scenario(
    required_actions=None,
    must_identify=None,
    must_not_miss=None,
    key_deadline=None,
    difficulty="medium",
):
    return {
        "scenario_id": "test-001",
        "difficulty":  difficulty,
        "ground_truth": {
            "correct_action_types": required_actions or ["reporting_change"],
            "key_deadline":         key_deadline,
        },
        "validation": {
            "must_identify":  must_identify or ["compliance deadline: 2026-03-31"],
            "must_act_on":    required_actions or ["reporting_change"],
            "must_not_miss":  must_not_miss or [],
        },
    }


def _make_state(
    classified_types=None,
    submitted_facts=None,
    key_deadline=None,
    completed=True,
):
    return {
        "scenario_id":      "test-001",
        "acknowledged":     True,
        "classified_types": classified_types or [],
        "submitted_facts":  submitted_facts or [],
        "escalated":        False,
        "clarifications":   [],
        "full_response": {
            "action_types":     classified_types or [],
            "identified_facts": submitted_facts or [],
            "key_deadline":     key_deadline,
            "penalty_summary":  None,
            "notes":            "",
        },
        "action_log": [],
        "completed":  completed,
    }


def test_pass_perfect():
    scenario = _make_scenario(
        required_actions=["reporting_change"],
        must_identify=["compliance deadline: 2026-03-31"],
        key_deadline="2026-03-31",
    )
    state = _make_state(
        classified_types=["reporting_change"],
        submitted_facts=["compliance deadline: 2026-03-31"],
        key_deadline="2026-03-31",
    )
    result = score_run(scenario, state)
    assert result.outcome == "PASS"
    assert result.action_coverage == 1.0
    assert result.deadline_accurate is True
    assert result.critical_misses == []


def test_fail_no_actions():
    scenario = _make_scenario(required_actions=["reporting_change", "process_change"])
    state    = _make_state(classified_types=[], submitted_facts=[])
    result   = score_run(scenario, state)
    assert result.outcome == "FAIL"
    assert result.action_coverage == 0.0


def test_partial_half_actions():
    scenario = _make_scenario(required_actions=["reporting_change", "process_change"])
    state    = _make_state(classified_types=["reporting_change"])
    result   = score_run(scenario, state)
    assert result.outcome == "PARTIAL"
    assert result.action_coverage == 0.5


def test_fail_critical_miss():
    scenario = _make_scenario(
        required_actions=["reporting_change"],
        must_not_miss=["penalty: TWD 1M per violation"],
    )
    state = _make_state(
        classified_types=["reporting_change"],
        submitted_facts=["some other fact"],
    )
    result = score_run(scenario, state)
    # action coverage is 1.0 but critical miss → FAIL
    assert result.outcome == "FAIL"
    assert len(result.critical_misses) > 0


def test_deadline_miss():
    scenario = _make_scenario(key_deadline="2026-03-31")
    state    = _make_state(
        classified_types=["reporting_change"],
        key_deadline=None,  # agent didn't provide deadline
    )
    result = score_run(scenario, state)
    assert result.deadline_accurate is False


def test_aggregate():
    scenario = _make_scenario()
    pass_state    = _make_state(
        classified_types=["reporting_change"],
        submitted_facts=["compliance deadline: 2026-03-31"],
    )
    fail_state = _make_state(classified_types=[], submitted_facts=[])

    results = [
        score_run(scenario, pass_state),
        score_run(scenario, fail_state),
    ]
    agg = aggregate_results(results)
    assert agg["total"] == 2
    assert agg["pass"] == 1
    assert agg["fail"] == 1
    assert agg["pass_rate"] == 0.5

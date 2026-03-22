"""
rab/scenarios/loader.py

Loads are_scenarios.json (produced by process.py scenarios) and
converts each entry into an ARE-compatible scenario dict.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_scenarios(
    path: str | Path,
    difficulty: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Load and optionally filter scenarios from are_scenarios.json.

    Args:
        path:       Path to are_scenarios.json.
        difficulty: If set, only return scenarios matching this difficulty
                    ('easy', 'medium', 'hard').
        limit:      If set, return at most this many scenarios.

    Returns:
        List of scenario dicts ready for simulation.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", [])

    if difficulty:
        scenarios = [s for s in scenarios if s.get("difficulty") == difficulty]

    if limit:
        scenarios = scenarios[:limit]

    return scenarios


def scenario_to_are_task(scenario: dict) -> dict:
    """Convert a RAB scenario dict into an ARE task specification.

    ARE expects a task with:
      - description: the situation + task for the agent
      - tools: list of available app methods
      - validation_fn: callable that checks agent state (injected at runtime)

    Returns:
        ARE-compatible task dict.
    """
    ctx = scenario.get("regulatory_context", {})
    situation  = scenario.get("situation", "")
    agent_task = scenario.get("agent_task", "")

    # Build the prompt the agent will receive
    prompt_lines = [
        f"REGULATORY UPDATE — {ctx.get('regulator', 'Unknown Regulator')}",
        f"Jurisdiction: {', '.join(ctx.get('jurisdiction', ['Unknown']))}",
        f"Update type: {ctx.get('update_type', 'Unknown')}",
        f"Published: {ctx.get('published_date', 'Unknown')}",
    ]
    if ctx.get("effective_date"):
        prompt_lines.append(f"Effective date: {ctx['effective_date']}")
    if ctx.get("compliance_deadline"):
        prompt_lines.append(f"Compliance deadline: {ctx['compliance_deadline']}")
    if ctx.get("source_url"):
        prompt_lines.append(f"Source: {ctx['source_url']}")

    prompt_lines += [
        "",
        "SITUATION:",
        situation,
        "",
        "YOUR TASK:",
        agent_task,
        "",
        "Use the compliance_workbench to:",
        "  1. acknowledge_change() — confirm you have read and understood the update",
        "  2. classify_required_action(action_types=[...]) — declare what actions are needed",
        "  3. submit_response(...) — submit your full compliance response",
        "  4. escalate() if the matter requires immediate leadership attention",
        "",
        "Valid action types: policy_change, process_change, reporting_change,",
        "  tech_data_change, training_change, escalate, acknowledge, other",
    ]

    return {
        "scenario_id":  scenario["scenario_id"],
        "title":        scenario.get("title", "Untitled"),
        "difficulty":   scenario.get("difficulty", "medium"),
        "prompt":       "\n".join(prompt_lines),
        "ground_truth": scenario.get("ground_truth", {}),
        "validation":   scenario.get("validation", {}),
        "tags":         scenario.get("tags", []),
        "metadata":     scenario.get("metadata", {}),
    }

#!/usr/bin/env python3
"""
run_simulation.py — RegulatoryAgentBench simulation runner

Uses LiteLLM for a single unified tool-use loop that works with any
model: OpenAI, Anthropic, Mistral, Llama, Gemini, etc.

LiteLLM normalises all provider responses to the OpenAI message format,
so the loop is identical regardless of which model is under evaluation.

Usage:
  python run_simulation.py run \\
    --scenarios are_scenarios.json \\
    --model gpt-5.4-mini-2026-03-17 \\
    --output results/gpt5mini.json

  python run_simulation.py run \\
    --scenarios are_scenarios.json \\
    --model claude-sonnet-4-20250514 \\
    --difficulty hard --limit 20

  python run_simulation.py compare \\
    --results results/gpt5mini.json \\
    --results results/claude.json
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import click
import litellm
from dotenv import load_dotenv

from rab.apps.compliance_workbench import ComplianceWorkbench
from rab.scenarios.loader import load_scenarios, scenario_to_are_task
from rab.scorer import ScoreResult, aggregate_results, score_run

load_dotenv(dotenv_path=Path(".env"), override=False)

litellm.suppress_debug_info = True

_DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"

# ---------------------------------------------------------------------------
# Tool definitions — OpenAI function-calling format (LiteLLM standard)
# LiteLLM translates these to each provider's native format automatically.
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "acknowledge_change",
            "description": "Acknowledge you have read and understood the regulatory update.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief acknowledgement text.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_required_action",
            "description": "Declare what type(s) of compliance action are required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of required action types.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why these actions are needed.",
                    },
                },
                "required": ["action_types"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_response",
            "description": "Submit your full compliance response. This completes the scenario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Final list of required action types.",
                    },
                    "identified_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key facts extracted from the regulatory update.",
                    },
                    "key_deadline": {
                        "type": "string",
                        "description": "Primary compliance deadline YYYY-MM-DD, or null.",
                    },
                    "penalty_summary": {
                        "type": "string",
                        "description": "Summary of consequences for non-compliance.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Any additional notes.",
                    },
                },
                "required": ["action_types", "identified_facts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": "Escalate the matter to legal/compliance leadership.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you are escalating.",
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["normal", "urgent", "critical"],
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_clarification",
            "description": (
                "Flag ambiguity and request clarification. "
                "No additional info will be provided — proceed with best available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"],
            },
        },
    },
]

_AGENT_SYSTEM = """\
You are a compliance agent at a regulated financial institution.

You will receive a regulatory update and must respond using the compliance_workbench tools.
Always follow this sequence:

1. Call acknowledge_change() — confirm you have read and understood the update.
2. Call classify_required_action() — declare what action types are needed.
3. Call submit_response() — submit your full compliance response including:
   - All required action types
   - Key facts you identified (deadlines, penalty amounts, specific requirements)
   - The primary compliance deadline (YYYY-MM-DD), if present
   - A brief penalty summary
4. Call escalate() if the matter requires immediate leadership attention.

Be specific. Surface dates, penalty amounts, and requirements explicitly.
Generic responses score poorly — fact extraction quality is evaluated.

Valid action types:
  policy_change, process_change, reporting_change, tech_data_change,
  training_change, escalate, acknowledge, other
"""


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _dispatch_tool(workbench: ComplianceWorkbench, name: str, arguments: str) -> str:
    """Parse arguments string and dispatch to the workbench. Returns JSON string."""
    try:
        kwargs = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid tool arguments: {exc}"})

    fn = getattr(workbench, name, None)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        return json.dumps(fn(**kwargs), ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Agent loop — single unified loop via LiteLLM
# ---------------------------------------------------------------------------

def run_agent_on_scenario(
    model:     str,
    are_task:  dict,
    max_turns: int = 10,
) -> tuple[dict, list[dict]]:
    """Run the agent on one scenario using LiteLLM's unified interface.

    LiteLLM normalises all provider responses to the OpenAI message format:
      response.choices[0].message.tool_calls  — always present, regardless of provider
      finish_reason == "tool_calls"            — model wants to call a tool
      finish_reason == "stop"                  — model is done

    This loop is identical for GPT, Claude, Mistral, Llama, Gemini, etc.
    LiteLLM handles the translation to each provider's native wire format.

    Returns:
        (workbench_state dict, turn_log list)
    """
    workbench = ComplianceWorkbench(scenario_id=are_task["scenario_id"])
    messages  = [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user",   "content": are_task["prompt"]},
    ]
    turn_log = []

    for turn in range(max_turns):
        response = litellm.completion(
            model    = model,
            messages = messages,
            tools    = _TOOLS,
        )

        choice     = response.choices[0]
        finish     = choice.finish_reason
        message    = choice.message
        tool_calls = message.tool_calls or []

        turn_entry = {
            "turn":       turn + 1,
            "finish":     finish,
            "tool_calls": [],
        }

        # Append assistant turn to history
        messages.append(message.model_dump())

        # No tool calls → model finished its response
        if finish == "stop" or not tool_calls:
            turn_log.append(turn_entry)
            break

        # Dispatch every tool call in this turn and collect results
        tool_results = []
        for tc in tool_calls:
            result_str = _dispatch_tool(workbench, tc.function.name, tc.function.arguments)
            turn_entry["tool_calls"].append({
                "tool":   tc.function.name,
                "input":  tc.function.arguments,
                "result": json.loads(result_str),
            })
            # Tool result goes back as a "tool" role message (OpenAI format)
            # LiteLLM translates this to the correct format for each provider
            tool_results.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result_str,
            })

        messages.extend(tool_results)
        turn_log.append(turn_entry)

        if workbench.state.completed:
            break

    return workbench.get_state(), turn_log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """RegulatoryAgentBench — simulate and compare AI compliance agent responses.\n

    \b
    Workflow:
      1. run     — run one model against scenarios, write results JSON
      2. compare — compare two or more result files side-by-side
    """


# ---------------------------------------------------------------------------
# Command: run
# ---------------------------------------------------------------------------

@cli.command("run")
@click.option(
    "--scenarios", "-s",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="are_scenarios.json produced by process.py scenarios.",
)
@click.option(
    "--model", "-m",
    default=_DEFAULT_MODEL,
    show_default=True,
    help=(
        "LiteLLM model string. Any supported provider works:\n"
        "  gpt-5.4-mini-2026-03-17        (default)\n"
        "  claude-sonnet-4-20250514\n"
        "  mistral/mistral-large-latest\n"
        "  groq/llama-3.3-70b-versatile\n"
        "Set the matching API key in .env."
    ),
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(dir_okay=False),
    help="Output JSON path. Defaults to results/<model>_<timestamp>.json.",
)
@click.option(
    "--difficulty",
    default=None,
    type=click.Choice(["easy", "medium", "hard"]),
    help="Filter scenarios to a specific difficulty tier.",
)
@click.option(
    "--limit", "-n",
    default=None,
    type=int,
    help="Run only the first N scenarios (useful for testing).",
)
@click.option(
    "--delay", "-d",
    default=1.0,
    show_default=True,
    type=float,
    help="Seconds to wait between scenario runs (rate-limit safety).",
)
@click.option(
    "--max-turns",
    default=10,
    show_default=True,
    type=int,
    help="Maximum agent turns per scenario.",
)
def run_cmd(scenarios, model, output, difficulty, limit, delay, max_turns):
    """Run one model against a scenario set and write a results JSON.\n

    Run this once per model you want to evaluate, then use 'compare'
    to produce a side-by-side summary across all runs.

    \b
    Example:
      python run_simulation.py run \\
        --scenarios are_scenarios.json \\
        --model gpt-5.4-mini-2026-03-17 \\
        --output results/gpt5mini.json
    """
    ts        = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_name = model.replace("/", "_").replace(":", "_")
    out_path  = Path(output) if output else Path("results") / f"{safe_name}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenario_list = load_scenarios(scenarios, difficulty=difficulty, limit=limit)
    if not scenario_list:
        raise click.UsageError("No scenarios found matching filters.")

    click.echo(f"\nModel      : {model}")
    click.echo(f"Scenarios  : {len(scenario_list)}")
    click.echo(f"Difficulty : {difficulty or 'all'}")
    click.echo(f"Output     : {out_path}\n")

    score_results: list[ScoreResult] = []
    run_details   = []

    for i, scenario in enumerate(scenario_list, 1):
        are_task = scenario_to_are_task(scenario)
        click.echo(
            f"[{i:3d}/{len(scenario_list)}] {are_task['difficulty'].upper():6s}  "
            f"{are_task['title'][:65]}",
            nl=False,
        )

        try:
            wb_state, turn_log = run_agent_on_scenario(model, are_task, max_turns)
            result = score_run(scenario, wb_state)
            score_results.append(result)

            icon = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗"}.get(result.outcome, "?")
            click.echo(
                f"  {icon} {result.outcome:7s}  "
                f"act:{result.action_coverage:.0%}  "
                f"facts:{result.fact_extraction:.0%}  "
                f"misses:{len(result.critical_misses)}"
            )

            run_details.append({
                "scenario_id":     scenario["scenario_id"],
                "title":           are_task["title"],
                "difficulty":      are_task["difficulty"],
                "outcome":         result.outcome,
                "scores": {
                    "action_coverage":   result.action_coverage,
                    "fact_extraction":   result.fact_extraction,
                    "deadline_accurate": result.deadline_accurate,
                    "critical_misses":   result.critical_misses,
                },
                "matched_actions":  result.matched_actions,
                "missing_actions":  result.missing_actions,
                "matched_facts":    result.matched_facts,
                "missing_facts":    result.missing_facts,
                "notes":            result.notes,
                "workbench_state":  wb_state,
                "turn_log":         turn_log,
            })

        except Exception as exc:
            click.echo(f"  ERROR: {exc}")
            run_details.append({
                "scenario_id": scenario.get("scenario_id"),
                "outcome":     "ERROR",
                "error":       str(exc),
            })

        if delay > 0 and i < len(scenario_list):
            time.sleep(delay)

    agg = aggregate_results(score_results)

    click.echo("\n" + "─" * 60)
    click.echo(
        f"  PASS {agg.get('pass', 0)}  "
        f"PARTIAL {agg.get('partial', 0)}  "
        f"FAIL {agg.get('fail', 0)}"
    )
    click.echo(f"  Pass rate    : {agg.get('pass_rate', 0):.1%}")
    click.echo(f"  Action cov   : {agg.get('avg_action_coverage', 0):.1%}")
    click.echo(f"  Fact extr    : {agg.get('avg_fact_extraction', 0):.1%}")
    click.echo(f"  Deadline acc : {agg.get('deadline_accuracy', 0):.1%}")

    out_path.write_text(
        json.dumps(
            {
                "run_at":    datetime.now(tz=timezone.utc).isoformat(),
                "model":     model,
                "scenarios": len(scenario_list),
                "aggregate": agg,
                "results":   run_details,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    click.echo(f"\n✓ Written → {out_path}")
    click.echo(
        f"  Next: python run_simulation.py compare "
        f"--results {out_path} --results <other>.json"
    )


# ---------------------------------------------------------------------------
# Command: compare
# ---------------------------------------------------------------------------

@cli.command("compare")
@click.option(
    "--results", "-r",
    "result_files",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Result JSON from a 'run'. Pass once per model.",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(dir_okay=False),
    help="Write comparison summary JSON to this path (optional).",
)
@click.option(
    "--show-failures",
    is_flag=True,
    default=False,
    help="Print per-scenario detail for FAIL outcomes.",
)
@click.option(
    "--show-consensus-failures",
    is_flag=True,
    default=False,
    help="Show scenarios where ALL models failed or only partially completed.",
)
def compare_cmd(result_files, output, show_failures, show_consensus_failures):
    """Compare results across multiple model runs side-by-side.\n

    Pass one --results flag per model run. Scenarios are aligned by
    scenario_id so each row in the comparison represents the same task.

    \b
    Example:
      python run_simulation.py compare \\
        --results results/gpt5mini.json \\
        --results results/claude.json \\
        --results results/mistral.json \\
        --show-failures
    """
    runs = []
    for path in result_files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        runs.append({
            "path":      path,
            "model":     data.get("model", Path(path).stem),
            "aggregate": data.get("aggregate", {}),
            "results":   {r["scenario_id"]: r for r in data.get("results", [])},
        })

    models    = [r["model"] for r in runs]
    col_w0    = min(max(len(m) for m in models) + 2, 40)

    # ── Aggregate table ──────────────────────────────────────────────────────
    click.echo(f"\n{'═' * 72}")
    click.echo("  AGGREGATE COMPARISON")
    click.echo(f"{'═' * 72}\n")

    cols    = ["Model", "Pass%", "P+Pa%", "ActCov", "FactEx", "Deadline", "AvgMiss"]
    widths  = [col_w0,  7,       7,       7,        7,        9,          8]

    click.echo("  " + "".join(h.ljust(w) for h, w in zip(cols, widths)))
    click.echo("  " + "─" * sum(widths))

    for run in runs:
        a = run["aggregate"]
        click.echo("  " + "".join([
            run["model"][:col_w0 - 1].ljust(widths[0]),
            f"{a.get('pass_rate', 0):.0%}".ljust(widths[1]),
            f"{a.get('pass_or_partial_rate', 0):.0%}".ljust(widths[2]),
            f"{a.get('avg_action_coverage', 0):.0%}".ljust(widths[3]),
            f"{a.get('avg_fact_extraction', 0):.0%}".ljust(widths[4]),
            f"{a.get('deadline_accuracy', 0):.0%}".ljust(widths[5]),
            f"{a.get('avg_critical_misses', 0):.2f}".ljust(widths[6]),
        ]))

    # ── By difficulty ────────────────────────────────────────────────────────
    click.echo(f"\n{'─' * 72}")
    click.echo("  PASS RATE BY DIFFICULTY\n")
    click.echo("  " + "Model".ljust(col_w0) + "Easy".ljust(8) + "Medium".ljust(8) + "Hard".ljust(8))
    click.echo("  " + "─" * (col_w0 + 24))

    for run in runs:
        counts: dict[str, dict] = defaultdict(lambda: {"total": 0, "pass": 0})
        for r in run["results"].values():
            d = r.get("difficulty", "unknown")
            counts[d]["total"] += 1
            if r.get("outcome") == "PASS":
                counts[d]["pass"] += 1

        def pct(d: str) -> str:
            c = counts.get(d, {})
            t = c.get("total", 0)
            return f"{c['pass'] / t:.0%}({t})" if t else "n/a"

        click.echo(
            "  " + run["model"][:col_w0 - 1].ljust(col_w0)
            + pct("easy").ljust(8)
            + pct("medium").ljust(8)
            + pct("hard").ljust(8)
        )

    # ── Most missed action types ─────────────────────────────────────────────
    click.echo(f"\n{'─' * 72}")
    click.echo("  MOST MISSED ACTION TYPES (top 5 per model)\n")

    for run in runs:
        miss_counts: dict[str, int] = defaultdict(int)
        for r in run["results"].values():
            for ma in r.get("missing_actions", []):
                miss_counts[ma] += 1
        if not miss_counts:
            continue
        top = sorted(miss_counts.items(), key=lambda x: -x[1])[:5]
        click.echo(f"  {run['model']}")
        for at, cnt in top:
            n = sum(run["results"][sid].get("outcome") is not None
                    for sid in run["results"])
            pct_str = f"{cnt / n:.0%}" if n else ""
            click.echo(f"    {cnt:4d} {pct_str:5s}  {at}")
        click.echo()

    # ── Per-scenario disagreements (only for exactly 2 runs) ─────────────────
    if len(runs) == 2:
        click.echo(f"{'─' * 72}")
        click.echo(f"  DISAGREEMENTS  [{runs[0]['model']}]  vs  [{runs[1]['model']}]\n")

        all_ids = set(runs[0]["results"]) | set(runs[1]["results"])
        disagree = []
        for sid in sorted(all_ids):
            r0 = runs[0]["results"].get(sid, {})
            r1 = runs[1]["results"].get(sid, {})
            o0, o1 = r0.get("outcome", "—"), r1.get("outcome", "—")
            if o0 != o1:
                disagree.append((r0.get("title") or sid, r0.get("difficulty", "?"), o0, o1))

        if disagree:
            click.echo(f"  {len(disagree)} scenarios where models gave different outcomes:\n")
            for title, diff, o0, o1 in disagree:
                click.echo(f"  [{diff[0].upper()}] {title[:58]}")
                click.echo(f"       {runs[0]['model'][:32]:32s} {o0}")
                click.echo(f"       {runs[1]['model'][:32]:32s} {o1}")
                click.echo()
        else:
            click.echo("  Models agreed on all scenarios.\n")

    # ── Failure detail ────────────────────────────────────────────────────────
    if show_failures:
        click.echo(f"{'─' * 72}")
        click.echo("  FAILURE DETAIL\n")
        for run in runs:
            fails = [r for r in run["results"].values() if r.get("outcome") == "FAIL"]
            if not fails:
                click.echo(f"  {run['model']} — no failures\n")
                continue
            click.echo(f"  {run['model']} — {len(fails)} failures")
            for r in fails:
                d = (r.get("difficulty") or "?")[0].upper()
                click.echo(f"    [{d}] {r.get('title', '')[:60]}")
                for ma in r.get("missing_actions", []):
                    click.echo(f"         missing action : {ma}")
                for cm in (r.get("scores") or {}).get("critical_misses", []):
                    click.echo(f"         critical miss  : {cm}")
            click.echo()


    # ── Consensus failures ───────────────────────────────────────────────────
    if show_consensus_failures:
        all_ids: set = set()
        for run in runs:
            all_ids |= set(run["results"].keys())

        consensus_fail: list = []    # all models FAIL
        consensus_partial: list = [] # no model PASS, at least one PARTIAL
        one_model_passes: list = []  # exactly one model passes

        for sid in sorted(all_ids):
            outcomes = {
                run["model"]: (run["results"].get(sid) or {}).get("outcome", "MISSING")
                for run in runs
            }
            n_pass    = sum(1 for o in outcomes.values() if o == "PASS")
            n_partial = sum(1 for o in outcomes.values() if o == "PARTIAL")

            title = next(
                (run["results"][sid].get("title") or sid
                 for run in runs if sid in run["results"]),
                sid,
            )
            diff = next(
                (run["results"][sid].get("difficulty", "?")
                 for run in runs if sid in run["results"]),
                "?",
            )
            entry = (sid, title, diff, outcomes)

            if n_pass == 0 and n_partial == 0:
                consensus_fail.append(entry)
            elif n_pass == 0:
                consensus_partial.append(entry)
            elif n_pass == 1:
                one_model_passes.append(entry)

        click.echo("\n" + "─" * 72)
        click.echo("  CONSENSUS FAILURES\n")
        click.echo(f"  All models FAILED          : {len(consensus_fail)}")
        click.echo(f"  No model fully PASSED      : {len(consensus_fail) + len(consensus_partial)}")
        click.echo(f"  Only one model passed      : {len(one_model_passes)}")

        if consensus_fail:
            click.echo(f"\n  ALL MODELS FAILED ({len(consensus_fail)} scenarios)\n")
            for sid, title, diff, outcomes in consensus_fail:
                d = (diff or "?")[0].upper()
                click.echo(f"  [{d}] {title[:65]}")
                for model_name, outcome in outcomes.items():
                    click.echo(f"       {model_name[:40]:40s} {outcome}")
                for run in runs:
                    r = run["results"].get(sid, {})
                    if r:
                        for ma in r.get("missing_actions", [])[:3]:
                            click.echo(f"         missing: {ma}")
                        break
                click.echo()

        if consensus_partial:
            click.echo(f"  NO MODEL FULLY PASSED ({len(consensus_partial)} scenarios)\n")
            for sid, title, diff, outcomes in consensus_partial:
                d = (diff or "?")[0].upper()
                click.echo(f"  [{d}] {title[:65]}")
                for model_name, outcome in outcomes.items():
                    click.echo(f"       {model_name[:40]:40s} {outcome}")
                click.echo()

        if one_model_passes:
            click.echo(f"  ONLY ONE MODEL PASSED ({len(one_model_passes)} scenarios)\n")
            for sid, title, diff, outcomes in one_model_passes:
                d = (diff or "?")[0].upper()
                winner = next(m for m, o in outcomes.items() if o == "PASS")
                short = winner.split("/")[-1][:20]
                click.echo(f"  [{d}] {title[:55]}  <- only {short}")
            click.echo()

    # ── Optional JSON output ──────────────────────────────────────────────────
    if output:
        comparison = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "models":       models,
            "aggregate":    {r["model"]: r["aggregate"] for r in runs},
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        click.echo(f"\n✓ Comparison written → {output}")


if __name__ == "__main__":
    cli()

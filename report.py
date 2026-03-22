#!/usr/bin/env python3
"""
report.py — RegulatoryAgentBench results analysis

Usage:
  python report.py --results results/run_01.json
  python report.py --results results/run_01.json --failures-only
  python report.py --results results/run_01.json --show-misses
"""

import json
from collections import defaultdict
from pathlib import Path

import click


@click.command("rab-report")
@click.option(
    "--results", "-r",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Results JSON from run_simulation.py.",
)
@click.option(
    "--failures-only",
    is_flag=True,
    help="Show only FAIL and ERROR scenarios.",
)
@click.option(
    "--show-misses",
    is_flag=True,
    help="Print critical_misses and missing_actions for each scenario.",
)
def cli(results, failures_only, show_misses):
    """Analyse and summarise a RegulatoryAgentBench simulation run."""
    data = json.loads(Path(results).read_text(encoding="utf-8"))

    agg      = data.get("aggregate", {})
    runs     = data.get("results", [])
    model    = data.get("model", "unknown")
    run_at   = data.get("run_at", "unknown")

    click.echo(f"\n{'═'*65}")
    click.echo(f"  RegulatoryAgentBench — Run Report")
    click.echo(f"{'═'*65}")
    click.echo(f"  Model    : {model}")
    click.echo(f"  Run at   : {run_at}")
    click.echo(f"  Scenarios: {data.get('scenarios', len(runs))}")
    click.echo(f"{'═'*65}\n")

    # Aggregate
    click.echo("AGGREGATE SCORES")
    click.echo(f"  Pass rate          : {agg.get('pass_rate', 0):.1%}  ({agg.get('pass',0)} / {agg.get('total',0)})")
    click.echo(f"  Pass+Partial rate  : {agg.get('pass_or_partial_rate', 0):.1%}")
    click.echo(f"  Avg action coverage: {agg.get('avg_action_coverage', 0):.1%}")
    click.echo(f"  Avg fact extraction: {agg.get('avg_fact_extraction', 0):.1%}")
    click.echo(f"  Deadline accuracy  : {agg.get('deadline_accuracy', 0):.1%}")
    click.echo(f"  Avg critical misses: {agg.get('avg_critical_misses', 0):.2f}")

    # By difficulty
    diff_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "pass": 0, "partial": 0, "fail": 0})
    for run in runs:
        d = run.get("difficulty", "unknown")
        diff_stats[d]["total"] += 1
        o = run.get("outcome", "FAIL")
        if o in diff_stats[d]:
            diff_stats[d][o.lower()] += 1

    click.echo("\nBY DIFFICULTY")
    for diff in ["hard", "medium", "easy", "unknown"]:
        s = diff_stats.get(diff)
        if not s or s["total"] == 0:
            continue
        pr = s["pass"] / s["total"]
        click.echo(
            f"  {diff:8s}: {s['total']:3d} scenarios  "
            f"pass:{s['pass']:3d} ({pr:.0%})  "
            f"partial:{s['partial']:3d}  fail:{s['fail']:3d}"
        )

    # Most missed action types
    miss_counts: dict[str, int] = defaultdict(int)
    for run in runs:
        for ma in run.get("missing_actions", []):
            miss_counts[ma] += 1

    if miss_counts:
        click.echo("\nMOST MISSED ACTION TYPES")
        for at, cnt in sorted(miss_counts.items(), key=lambda x: -x[1]):
            click.echo(f"  {cnt:4d}  {at}")

    # Per-scenario
    click.echo(f"\n{'─'*65}")
    click.echo("PER-SCENARIO RESULTS\n")

    icon_map = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗", "ERROR": "!"}
    for run in runs:
        outcome = run.get("outcome", "FAIL")
        if failures_only and outcome in ("PASS", "PARTIAL"):
            continue

        icon   = icon_map.get(outcome, "?")
        sc     = run.get("scores", {})
        title  = (run.get("title") or run.get("scenario_id", ""))[:60]
        diff   = run.get("difficulty", "?")[:1].upper()

        click.echo(
            f"  {icon} [{diff}] {title}\n"
            f"       actions:{sc.get('action_coverage',0):.0%}  "
            f"facts:{sc.get('fact_extraction',0):.0%}  "
            f"deadline:{sc.get('deadline_accurate','?')}  "
            f"critical_misses:{len(sc.get('critical_misses',[]))}"
        )

        if show_misses:
            for ma in run.get("missing_actions", []):
                click.echo(f"         missing action : {ma}")
            for mf in run.get("missing_facts", []):
                click.echo(f"         missing fact   : {mf}")
            for cm in sc.get("critical_misses", []):
                click.echo(f"         critical miss  : {cm}")

        click.echo()

    click.echo(f"{'═'*65}")
    click.echo(f"  Full results: {results}\n")


if __name__ == "__main__":
    cli()

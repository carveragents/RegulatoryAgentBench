# RegulatoryAgentBench (RAB)

**A benchmark for evaluating AI agent responses to real-world regulatory change.**

Built on Meta's [ARE (Agent Research Environments)](https://github.com/facebookresearch/meta-agents-research-environments), RAB turns annotated regulatory updates into dynamic compliance scenarios — testing whether AI agents can not just *understand* regulatory change, but *act on it correctly*.

---

## Why this exists

Most agent benchmarks use static, synthetic tasks. Regulatory compliance is neither.  
Rules change on specific dates. Deadlines cascade. Jurisdictions conflict. Penalties are real.

RAB evaluates agents on scenarios derived from **real regulatory updates** across hundreds of global regulators — grounded in structured annotations that encode what changed, what action is required, by when, and what happens if you miss it.

---

## How it works

```
Regulatory annotations  →  process.py select   →  selected_artifacts.json
                        →  process.py scenarios →  are_scenarios.json
                        →  run_simulation.py    →  results/
```

1. **`process.py select`** (in your private annotation pipeline) filters 100K+ raw annotations down to high-signal simulation candidates — artifacts with clear actionables, compliance dates, and defined consequences.

2. **`process.py scenarios`** calls an LLM to convert each annotation into a structured ARE scenario with `required_actions`, `validation`, and `ground_truth`.

3. **`run_simulation.py`** (this repo) loads scenarios into ARE, runs your agent against each one, and scores responses against ground truth.

---

## Scenario structure

Each scenario in `are_scenarios.json` looks like:

```json
{
  "scenario_id": "...",
  "title": "FSC Taiwan: New reporting requirement for fund managers",
  "difficulty": "medium",
  "regulatory_context": {
    "regulator": "金融監督管理委員會",
    "jurisdiction": ["TW"],
    "update_type": "reporting_requirement",
    "compliance_deadline": "2026-03-31"
  },
  "situation": "...",
  "agent_task": "...",
  "required_actions": [...],
  "validation": {
    "must_identify": ["compliance deadline: 2026-03-31"],
    "must_act_on": ["reporting_change"],
    "must_not_miss": ["Q1 filing deadline", "penalty: TWD 1M per violation"]
  },
  "ground_truth": {
    "correct_action_types": ["reporting_change", "process_change"],
    "key_deadline": "2026-03-31",
    "penalty_summary": "..."
  }
}
```

---

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Install and configure ARE
pip install meta-agents-research-environments
cp example.env .env        # add your LLM API key

# 3. Run simulation
python run_simulation.py \
  --scenarios are_scenarios/example_scenarios.json \
  --agent default \
  --output results/run_01.json

# 4. View results
python report.py --results results/run_01.json
```

---

## Installation

**Prerequisites:** Python 3.10+, an Anthropic or compatible LLM API key.

```bash
git clone https://github.com/your-org/RegulatoryAgentBench
cd RegulatoryAgentBench
pip install -e .
```

---

## Repository structure

```
RegulatoryAgentBench/
├── rab/
│   ├── apps/
│   │   └── compliance_workbench.py   # ARE app: the compliance action surface
│   ├── scenarios/
│   │   └── loader.py                 # loads are_scenarios.json into ARE
│   ├── scorer.py                     # validation + scoring logic
│   └── __init__.py
├── are_scenarios/
│   └── example_scenarios.json        # small example set (5 scenarios)
├── results/                          # simulation run outputs
├── tests/
│   └── test_scorer.py
├── run_simulation.py                 # main entry point
├── report.py                         # results analysis and summary
├── example.env
├── pyproject.toml
└── README.md
```

---

## Scoring

Each scenario is scored on:

| Dimension | Description |
|---|---|
| **Action coverage** | Did the agent identify all `must_act_on` action types? |
| **Fact extraction** | Did the agent surface all `must_identify` facts? |
| **Critical miss** | Did the agent miss anything in `must_not_miss`? (-1 penalty) |
| **Deadline accuracy** | Did the agent correctly identify `key_deadline`? |

A scenario is **passed** if action coverage ≥ 1.0 and no critical misses.  
A scenario is **partially passed** if action coverage ≥ 0.5 and ≤ 1 critical miss.

---

## Difficulty tiers

| Tier | Criteria | Expected agent pass rate |
|---|---|---|
| **Hard** | High impact + high urgency, multiple jurisdictions | ~40–60% |
| **Medium** | Medium impact or urgency, clear actionables | ~60–80% |
| **Easy** | Low urgency, single jurisdiction, explicit requirements | ~80–95% |

---

## License

MIT. See [LICENSE](LICENSE).

---

## Citation

If you use RAB in your research:

```bibtex
@misc{rab2026,
  title  = {RegulatoryAgentBench: Evaluating AI Agent Responses to Regulatory Change},
  year   = {2026},
  url    = {https://github.com/your-org/RegulatoryAgentBench}
}
```

Built on [Meta ARE](https://arxiv.org/abs/2509.17158).

# RegulatoryAgentBench (RAB)

**A benchmark for evaluating AI agent responses to real-world regulatory change.**

RAB turns annotated regulatory updates into dynamic compliance scenarios — testing whether AI agents can not just *understand* regulatory change, but *act on it correctly*.

> **Design note:** RAB is inspired by Meta's [ARE (Agent Research Environments)](https://github.com/facebookresearch/meta-agents-research-environments), which introduced the idea of evolving, multi-step agent evaluation environments. RAB applies that framing to regulatory compliance: scenarios derived from real regulatory updates, with structured ground truth, typed actions, and a compliance workbench the agent must use to respond. RAB is a standalone implementation and does not depend on the ARE codebase.

---

## Why this exists

Most agent benchmarks use static, synthetic tasks. Regulatory compliance is neither.  
Rules change on specific dates. Deadlines cascade. Jurisdictions conflict. Penalties are real.

RAB evaluates agents on scenarios derived from **real regulatory updates** across hundreds of global regulators — grounded in structured annotations that encode what changed, what action is required, by when, and what happens if you miss it.

---

## Results

Benchmarked across 50 scenarios from 30+ regulators across APAC, EMEA, LATAM, and North America.

Full benchmark results: [results/summary.txt](results/summary.txt)

### Aggregate scores

| Model | Pass% | Pass+Partial% | Action Cov | Fact Extr | Deadline Acc |
|---|---|---|---|---|---|
| gpt-5-mini | 74% | 98% | 94% | 33% | 96% |
| gpt-5.4 | 64% | 96% | 87% | 31% | 98% |
| gpt-5.4-mini | 48% | 94% | 82% | 26% | 98% |

### Pass rate by difficulty

| Model | Easy (15) | Medium (17) | Hard (18) |
|---|---|---|---|
| gpt-5-mini | 67% | 59% | **94%** |
| gpt-5.4 | 67% | 59% | 67% |
| gpt-5.4-mini | 47% | 35% | 61% |

### Most missed action types (all models)

| Action type | gpt-5.4 | gpt-5.4-mini | gpt-5-mini |
|---|---|---|---|
| tech_data_change | 12% | 16% | 10% |
| policy_change | 10% | 10% | — |
| reporting_change | 8% | 10% | — |
| other_change | 6% | 6% | 6% |

### What the results show

**The headline pass rates are only part of the story.** All three models score 94–98% on Pass+Partial, meaning they almost always get *something* right. The real differentiation is in how completely they respond.

**Deadline accuracy is a solved problem** (96–98% across all models). Agents reliably identify compliance dates. This is no longer a meaningful differentiator.

**Fact extraction is the unsolved problem** (26–33% across all models). Agents understand that action is needed but consistently fail to surface the specific facts a compliance team requires — penalty amounts, registration consequences, downstream system requirements. A response that identifies the right action type but omits the penalty detail is operationally useless.

**`tech_data_change` is the universal blind spot.** Every model misses it most often. Agents understand the policy intent of a regulatory change but consistently fail to identify its downstream implications for systems and data. This is the gap between reading a regulation and understanding what it means for your infrastructure.

**The CMA Auditor Registration scenario failed all three models** — the only consensus failure. The critical misses (`registration_expiry_consequences`, `database_update_requirement`) reveal a specific pattern: when consequences are implicit in the regulatory context rather than stated explicitly in the update, all models fail to surface them.

**gpt-5-mini's hard scenario performance (94%) looks strong but is partly an artifact of decisiveness.** It passed 9 of the 11 "only one model passed" scenarios — but in each case it was the *only* model that passed, meaning it commits to answers more readily rather than being consistently more accurate. On easy scenarios it performs identically to gpt-5.4 (67%).

**10 scenarios had no model fully pass** — these are the most valuable for understanding agent capability limits. They cluster around scenarios requiring multi-step implicit reasoning: connecting a regulatory change to its operational consequences, not just its stated requirements.

---

## How it works

```
Regulatory annotations  →  process.py select    →  selected_artifacts.json
                        →  process.py scenarios  →  are_scenarios.json
                        →  run_simulation.py run  →  results/<model>.json
                        →  run_simulation.py compare → side-by-side report
```

1. **`process.py select`** filters 100K+ raw annotations down to high-signal simulation candidates — artifacts with clear actionables, compliance dates, and defined consequences. Stratified across impact × urgency × geography to avoid regional bias.

2. **`process.py scenarios`** calls an LLM to convert each annotation into a structured scenario with `required_actions`, `validation`, and `ground_truth`.

3. **`run_simulation.py run`** runs a single model against the scenario set via LiteLLM — works with OpenAI, Anthropic, Mistral, Llama, and any other LiteLLM-supported provider.

4. **`run_simulation.py compare`** produces a side-by-side report including consensus failure analysis: scenarios where all models failed, no model fully passed, or only one model passed.

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
# 1. Clone and install
git clone https://github.com/your-org/RegulatoryAgentBench
cd RegulatoryAgentBench
pip install -r requirements.txt

# 2. Configure
cp example.env .env   # add your API key(s)

# 3. Run one model
python run_simulation.py run \
  --scenarios are_scenarios/scenarios.json \
  --model gpt-5.4-mini-2026-03-17 \
  --output results/gpt5mini.json

# 4. Run another
python run_simulation.py run \
  --scenarios are_scenarios/scenarios.json \
  --model claude-sonnet-4-20250514 \
  --output results/claude.json

# 5. Compare with consensus failure analysis
python run_simulation.py compare \
  --results results/gpt5mini.json \
  --results results/claude.json \
  --show-consensus-failures
```

---

## Repository structure

```
RegulatoryAgentBench/
├── rab/
│   ├── apps/
│   │   └── compliance_workbench.py   # action surface exposed to the agent
│   ├── scenarios/
│   │   └── loader.py                 # loads are_scenarios.json
│   ├── scorer.py                     # validation + scoring logic
│   └── __init__.py
├── are_scenarios/                    # your scenario JSON files go here
├── results/                          # simulation run outputs
├── tests/
│   └── test_scorer.py
├── run_simulation.py                 # main entry point (run + compare)
├── report.py                         # single-run results analysis
├── requirements.txt
├── example.env
└── README.md
```

---

## Scoring

Each scenario is scored on four dimensions:

| Dimension | Description |
|---|---|
| **Action coverage** | Did the agent identify all `must_act_on` action types? |
| **Fact extraction** | Did the agent surface all `must_identify` facts? |
| **Deadline accuracy** | Did the agent correctly identify `key_deadline`? |
| **Critical miss** | Did the agent miss anything in `must_not_miss`? |

**PASS** — action coverage = 100% and zero critical misses  
**PARTIAL** — action coverage ≥ 50% and ≤ 1 critical miss  
**FAIL** — everything else

---

## Difficulty tiers

| Tier | Criteria |
|---|---|
| **Hard** | High impact + high urgency, multiple jurisdictions or conflicting requirements |
| **Medium** | Medium impact or urgency, clear but non-trivial actionables |
| **Easy** | Low urgency, single jurisdiction, explicit requirements |

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

Inspired by [Meta ARE](https://arxiv.org/abs/2509.17158).


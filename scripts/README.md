# Scripts

```pip install git+https://github.com/sunblaze-ucb/progent.git```


## Sliding Window Evaluation

Three scenarios for evaluating utility and attack success rate:

| Script | Scenario | Exposure | Defense | Metrics |
|--------|----------|----------|---------|---------|
| `sliding_window_utility.sh` | Utility baseline | None | None (`raw`) | Task completion score |
| `sliding_window_utility_defense.sh` | Utility + instruction defense | None | `instructional` prompt guard | Task completion — utility cost of instruction defense |
| `sliding_window_progent.sh` | Utility + Progent | None | Progent `static` | Task completion — utility cost of Progent |
| `sliding_window_drift.sh` | Utility + DRIFT | None | DRIFT (all 3 stages) | Task completion — utility cost of DRIFT |
| `sliding_window.sh` | Attack, no defense | Yes | None (`raw`) | ASR + task completion |
| `sliding_window_attack_progent.sh` | Attack + Progent | Yes | Progent `static` | ASR + task completion |
| `sliding_window_attack_drift.sh` | Attack + DRIFT | Yes | DRIFT (all 3 stages) | ASR + task completion |

**Judge model**: `openai/gpt-5-mini` (default for all)

**Progent** (`--progent_guard 1`):
- `--progent_guard_mode static`: fixed policy — forbids `execute_command`, allows `search`/`read`
- `--progent_guard_mode dynamic`: LLM generates per-task policy from user query

**DRIFT** (`--drift_guard 1`):
- Stage 1 (`--drift_build_constraints 1`): Secure Planner — LLM builds expected tool trajectory + parameter checklist before execution
- Stage 2 (`--drift_dynamic_validation 1`): Dynamic Validator — validates each tool call against planned trajectory; blocks unplanned Write/Execute calls
- Stage 3 (`--drift_injection_isolation 1`): Injection Isolator — LLM detects injected instructions in tool results and masks them before they enter memory
- `--drift_guard_llm_name`: separate small LLM for DRIFT's internal calls (e.g., `openai/gpt-4.1-nano`); defaults to agent model

**Save dirs** (under `/data2/xianglin/zombie_agent/sliding_window/`):
- `utility_clean/` — baseline
- `utility_defense/` — instruction guard
- `utility_progent/` — Progent
- `utility_drift/` — DRIFT
- `attack/` — attack, no defense
- `attack_progent/` — attack + Progent
- `attack_drift/` — attack + DRIFT

---

## RAG Evaluation

| Script | Description |
|--------|-------------|
| `rag_exposure.sh` | Exposure phase only — injects payloads into RAG memory |
| `rag_trigger.sh` | Trigger phase only — assumes exposure already in DB |
| `rag.sh` | Both phases (exposure + trigger) |

### Adaptive Attack (RAG)

| Script | Description |
|--------|-------------|
| `rag_adaptive_exposure_only.sh` | Adaptive exposure only |
| `rag_adaptive_trigger_only.sh` | Adaptive trigger only |
| `rag_adaptive.sh` | Full adaptive attack (both phases) |
| `rag_adaptive_compare_controllers.sh` | Compare different adaptive attack controllers |

---

## Other

| Script | Description |
|--------|-------------|
| `RAG_build_corpus.sh` | Build base corpus for RAG memory (ChromaDB) |
| `sl_adaptive.sh` | Adaptive attack on sliding window agent |
| `plot.sh` | Generate plots from saved result files |

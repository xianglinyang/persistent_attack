# Scripts

## Sliding Window Evaluation

Three scenarios for evaluating utility and attack success rate:

| Script | Scenario | Exposure | Defense (`instruction_guard`) | Metrics |
|--------|----------|----------|-------------------------------|---------|
| `sliding_window_utility.sh` | Utility baseline | None (`exposure_rounds=0`) | `raw` (off) | Task completion score |
| `sliding_window_utility_defense.sh` | Utility under defense | None (`exposure_rounds=0`) | `instructional` (on) | Task completion score — shows utility cost of defense |
| `sliding_window.sh` | Attack | Yes (`exposure_rounds=3`) | `raw` (off) | ASR + task completion — shows attack impact |

**Judge model**: `openai/gpt-5-mini` (default for all three)

**Save dirs** (under `/data2/xianglin/zombie_agent/sliding_window/`):
- `utility_clean/`
- `utility_defense/`
- `attack/`

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

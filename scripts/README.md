# Scripts

```bash
pip install git+https://github.com/sunblaze-ucb/progent.git
```

---

## Models

Switch the active model by editing `MODEL_NAME` / `ABBR` at the top of each script.

| ABBR | MODEL_NAME |
|------|-----------|
| `glm` | `z-ai/glm-4.7-flash` |
| `gemini` | `google/gemini-2.5-flash` |
| `ds` | `deepseek/deepseek-v3.2` |
| `llama` | `meta-llama/llama-3.3-70b-instruct` |
| `qwen` | `qwen/qwen3-235b-a22b` |

> **Note:** PIGuard and ProtectAIv2 require local HTTP services before running:
> - PIGuard → port `12390`
> - ProtectAIv2 → port `12392`

---

## Sliding Window

### Step 1 — Utility baseline (no attack, no defense)

Fills the **utility / None** row.

```bash
# Repeat for each model (glm / gemini / ds / llama / qwen)
bash scripts/sliding_window_utility.sh
```

5 runs total.

---

### Step 2 — Utility with defense (no attack, defense on)

Fills the **utility** column for all 7 defense rows.

start the guard model service

```bash
# PIGuard
cd /home/xianglin/git_space/persistent_attack/src/guard/detection_based/PIGuard
uvicorn app:app --host 127.0.0.1 --port 12390

# PromptGuard
cd /home/xianglin/git_space/persistent_attack/src/guard/detection_based/PromptGuard
uvicorn app:app --host 127.0.0.1 --port 12391

# AIProtectv2
cd /home/xianglin/git_space/persistent_attack/src/guard/detection_based/ProtectAIv2
CUDA_VISIBLE_DEVICES=1 uvicorn app:app --host 127.0.0.1 --port 12392
```

```bash
# Repeat for each model
bash scripts/sliding_window_utility_defense.sh
```

Each run executes 7 configurations internally:
`PIGuard` · `ProtectAIv2` · `sandwich` · `spotlight` · `instructional` · `Progent dynamic` · `DRIFT`

5 runs total.

---

### Step 3 — Attack + all defenses

Fills the **ASR** column for all rows (baseline + 7 defenses).

```bash
# Repeat for each model
bash scripts/sliding_window.sh
```

Each run executes 8 configurations internally (baseline + 7 defenses).

5 runs total.

---

### Save dirs (Sliding Window)

| Data | Path |
|------|------|
| Utility baseline | `/data2/xianglin/zombie_agent/sliding_window/utility_clean/` |
| Utility + defense | `/data2/xianglin/zombie_agent/sliding_window/utility_defense/` |
| Attack + defense | `/data2/xianglin/zombie_agent/sliding_window/` |

Result filenames encode all key parameters (model, method, attack_type, detection_guard, guard_model, instruction_guard) so results from different models and defenses never overwrite each other.

---

## RAG

### Step 0 — Build base corpus (run once for all models)

Populates the `base` ChromaDB collection with MS MARCO passages.
**Run this once before any RAG experiment.**

```bash
bash scripts/RAG_build_corpus.sh
```

Builds 5 DBs in one run (one per model), stored at:
`/data2/xianglin/zombie_agent/{ABBR}_zombie_completion_real_db/msmarco/`

---

### Step 1 — Utility baseline (no attack, no defense)

Fills the **utility / None** row for RAG.

```bash
# Repeat for each model
bash scripts/rag_utility.sh
```

5 runs total.

---

### Step 2 — Utility with defense (no attack, defense on)

Fills the **utility** column for all 7 defense rows.

```bash
# Repeat for each model
bash scripts/rag_utility_defense.sh
```

Each run executes 7 configurations internally.

5 runs total.

---

### Step 3 — Exposure (inject payload, run once per model)

Injects the malicious payload into the RAG vector database.
**Must complete before step 4.**

```bash
# Repeat for each model
bash scripts/rag_exposure.sh
```

5 runs total.

---

### Step 4 — Attack + all defenses

Fills the **ASR** column for all rows (baseline + 7 defenses).

```bash
# Repeat for each model (requires exposure DB from step 3)
bash scripts/rag_trigger.sh
```

Each run executes 8 configurations internally.

5 runs total.

---

### Save dirs (RAG)

| Data | Path |
|------|------|
| Utility baseline | `/data2/xianglin/zombie_agent/rag/utility_clean/` |
| Utility + defense | `/data2/xianglin/zombie_agent/rag/utility_defense/` |
| Attack + defense | `/data2/xianglin/zombie_agent/{ABBR}_{METHOD}_{ATTACK_TYPE}_db/msmarco/log_results/` |

> **Why utility reuses the attack DB:** `exposure_rounds=0` makes retrieval skip the exposure
> collection entirely (all injected docs have round ≥ 1), so the agent only sees clean base docs.

---

## Full Run Order

```bash
# ── For each of the 5 models ──────────────────────────────────

# Sliding Window
bash scripts/sliding_window_utility.sh           # utility baseline
bash scripts/sliding_window_utility_defense.sh   # utility + 7 defenses
bash scripts/sliding_window.sh                   # attack + 7 defenses

# RAG
bash scripts/RAG_build_corpus.sh                 # build base corpus (once, all models)
bash scripts/rag_utility.sh                      # utility baseline
bash scripts/rag_utility_defense.sh              # utility + 7 defenses
bash scripts/rag_exposure.sh                     # inject payload (once per model)
bash scripts/rag_trigger.sh                      # attack + 7 defenses
```

**Total: 5 models × 5 scripts = 25 runs**

---

## Other Scripts

| Script | Description |
|--------|-------------|
| `RAG_build_corpus.sh` | Build base corpus for RAG memory (ChromaDB) |
| `sl_adaptive.sh` | Adaptive attack on sliding window agent |
| `rag_adaptive.sh` | Full adaptive RAG attack (exposure + trigger) |
| `rag_adaptive_exposure_only.sh` | Adaptive RAG — exposure only |
| `rag_adaptive_trigger_only.sh` | Adaptive RAG — trigger only |
| `rag_adaptive_compare_controllers.sh` | Compare adaptive attack controllers |
| `plot.sh` | Generate plots from saved result files |

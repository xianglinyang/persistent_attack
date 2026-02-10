# RAG Adaptive Attack Scripts

This directory contains bash scripts for running adaptive attacks on RAG agents with different search-based controllers (PAIR, MAP-Elites, TAP, AutoDAN).

## Available Scripts

### 1. `rag_adaptive.sh` - Main Script (All-in-One)
**Purpose**: Run complete adaptive attack with both exposure and trigger phases.

**Features**:
- Supports all 4 controllers: PAIR, MAP-Elites, TAP, AutoDAN
- Configurable phase selection: exposure, trigger, or both
- Full parameter control

**Usage**:
```bash
# Edit configuration in the script, then run:
./scripts/rag_adaptive.sh

# Or customize inline:
PHASE="both" CONTROLLER_TYPE="pair" ./scripts/rag_adaptive.sh
```

**Key Parameters**:
- `PHASE`: "exposure", "trigger", or "both" (default: "both")
- `CONTROLLER_TYPE`: "pair", "map_elites", "tap", or "autodan" (default: "pair")
- `EXPOSURE_ROUNDS`: Number of exposure rounds (default: 3)
- `TRIGGER_ROUNDS`: Number of trigger rounds (default: 50)
- `BUDGET_PER_ROUND`: Optimization budget per round (default: 10)
- `TOTAL_BUDGET`: Total optimization budget (default: 20)

---

### 2. `rag_adaptive_exposure_only.sh` - Exposure Phase Only
**Purpose**: Build attack database by running only the exposure phase.

**Use Case**: 
- Initial setup: Create exposure data
- Testing exposure optimization
- Building reusable attack database

**Usage**:
```bash
./scripts/rag_adaptive_exposure_only.sh
```

**After Running**: 
The exposure data is saved to the RAG database. You can run multiple trigger experiments on this data.

---

### 3. `rag_adaptive_trigger_only.sh` - Trigger Phase Only
**Purpose**: Run trigger phase using existing exposure data.

**Use Case**:
- Testing different datasets on same exposure
- Multiple trigger evaluations
- Quick attack validation

**Usage**:
```bash
# Make sure EXPOSURE_ROUNDS matches your exposure experiment
./scripts/rag_adaptive_trigger_only.sh
```

**Prerequisites**: 
- Exposure phase must be completed first
- `EXPOSURE_ROUNDS` must match the number of rounds from exposure phase

---

### 4. `rag_adaptive_compare_controllers.sh` - Controller Comparison
**Purpose**: Run all 4 controllers sequentially for comparison.

**Controllers Tested**:
1. PAIR (Prompt Automatic Iterative Refinement)
2. MAP-Elites (Quality-Diversity)
3. TAP (Tree of Attacks with Pruning)
4. AutoDAN (Genetic Algorithm)

**Usage**:
```bash
./scripts/rag_adaptive_compare_controllers.sh
```

**Output Structure**:
```
results/rag_controller_comparison/
├── pair/
│   ├── metrics_exposure_*.json
│   ├── metrics_trigger_*.json
│   └── attack_*.png
├── map_elites/
├── tap/
└── autodan/
```

---

## Configuration Parameters

### Model Settings
```bash
MODEL_NAME="google/gemini-2.5-flash"          # Target model
ATTACKER_MODEL_NAME="openai/gpt-4o-mini"      # Attacker model for optimization
```

### RAG-Specific Settings
```bash
DB_PATH="/data2/xianglin/zombie_agent/db_storage/msmarco"
TOP_K=20                    # Number of documents to retrieve
EVOLVE_MODE="raw"           # "raw", "reflection", or "experience"
RESET=1                     # 1: Reset DB, 0: Resume from existing
BASE_NAME="base"            # Base collection name
EXPOSURE_NAME="exposure"    # Exposure collection name
TRIGGER_NAME="trigger"      # Trigger collection name
```

### Attack Budget
```bash
EXPOSURE_ROUNDS=3           # Number of exposure rounds
TRIGGER_ROUNDS=50           # Number of trigger rounds
BUDGET_PER_ROUND=10         # Optimization steps per round
TOTAL_BUDGET=20             # Total optimization budget
```

### Guard Settings
```bash
DETECTION_GUARD=0           # 0: Disabled, 1: Enabled
DETECTION_GUARD_MODEL_NAME="openai/gpt-4.1-nano"
INSTRUCTION_GUARD_NAME="raw"  # "raw", "sandwich", "instructional", etc.
```

### Controller-Specific (AutoDAN)
```bash
AUTODAN_POPULATION_SIZE=20  # Population size for genetic algorithm
AUTODAN_NUM_ELITES=2        # Number of elite individuals to preserve
AUTODAN_MUTATION_RATE=0.1   # Mutation rate (0.0-1.0)
AUTODAN_CROSSOVER_RATE=0.5  # Crossover rate (0.0-1.0)
```

---

## Common Workflows

### Workflow 1: Quick Test
```bash
# Run PAIR attack with minimal settings
EXPOSURE_ROUNDS=2 TRIGGER_ROUNDS=5 TOTAL_BUDGET=10 ./scripts/rag_adaptive.sh
```

### Workflow 2: Compare All Controllers
```bash
# Run all 4 controllers for comparison
./scripts/rag_adaptive_compare_controllers.sh

# Analyze results
python -m src.analysis.compare_controllers --base_dir results/rag_controller_comparison
```

### Workflow 3: Reuse Exposure Data
```bash
# Step 1: Build exposure database
./scripts/rag_adaptive_exposure_only.sh

# Step 2: Run multiple trigger experiments
DATASET_NAME_OR_PATH="dataset1" ./scripts/rag_adaptive_trigger_only.sh
DATASET_NAME_OR_PATH="dataset2" ./scripts/rag_adaptive_trigger_only.sh
DATASET_NAME_OR_PATH="dataset3" ./scripts/rag_adaptive_trigger_only.sh
```

### Workflow 4: Test with Guards
```bash
# Enable detection guard
DETECTION_GUARD=1 DETECTION_GUARD_MODEL_NAME="openai/gpt-5-nano" ./scripts/rag_adaptive.sh

# Test different instruction guards
INSTRUCTION_GUARD_NAME="sandwich" ./scripts/rag_adaptive.sh
INSTRUCTION_GUARD_NAME="instructional" ./scripts/rag_adaptive.sh
```

### Workflow 5: Different Controllers
```bash
# Test PAIR
CONTROLLER_TYPE="pair" ./scripts/rag_adaptive.sh

# Test MAP-Elites
CONTROLLER_TYPE="map_elites" BUDGET_PER_ROUND=15 ./scripts/rag_adaptive.sh

# Test TAP
CONTROLLER_TYPE="tap" ./scripts/rag_adaptive.sh

# Test AutoDAN with custom settings
CONTROLLER_TYPE="autodan" \
AUTODAN_POPULATION_SIZE=30 \
AUTODAN_NUM_ELITES=3 \
./scripts/rag_adaptive.sh
```

---

## Troubleshooting

### Issue: "Database not found"
**Solution**: Check `DB_PATH` points to correct location or set `RESET=1` to create new database.

### Issue: "No exposure data found" (in trigger-only mode)
**Solution**: Run exposure phase first, or check `EXPOSURE_ROUNDS` matches your exposure experiment.

### Issue: Script permission denied
**Solution**: 
```bash
chmod +x scripts/rag_adaptive*.sh
```

### Issue: Out of memory
**Solution**: Reduce `AUTODAN_POPULATION_SIZE` or `BUDGET_PER_ROUND`

---

## Output Structure

Each script saves results to the specified `SAVE_DIR`:

```
results/
└── search_based_rag/
    ├── metrics_exposure_*.json      # Exposure phase metrics
    ├── metrics_trigger_*.json       # Trigger phase metrics
    └── attack_*.png                 # Visualization plots
```

**Metrics Files**:
- Exposure: payload_count, recall@k (if triggered during exposure)
- Trigger: payload_count, recall@10/20/50, ASR (exfiltration, command_exec, reload_times)

---

## Comparison with sl_adaptive.sh

| Feature | sl_adaptive.sh | rag_adaptive.sh |
|---------|----------------|-----------------|
| Memory Type | Sliding Window | RAG (Vector DB) |
| Metrics | Payload in window | Payload count, Recall@k |
| DB Management | N/A | Reset/Resume |
| Parameters | `WINDOW_SIZE` | `DB_PATH`, `TOP_K`, `EVOLVE_MODE` |
| Collections | N/A | `BASE_NAME`, `EXPOSURE_NAME`, `TRIGGER_NAME` |

---

## Tips for Best Results

1. **Budget Allocation**: Set `TOTAL_BUDGET` = `EXPOSURE_ROUNDS` × `BUDGET_PER_ROUND` + `TRIGGER_ROUNDS` × `BUDGET_PER_ROUND`
2. **Controller Selection**: 
   - PAIR: Fast, good for simple scenarios
   - MAP-Elites: Best for diversity exploration
   - TAP: Good for tree-based search
   - AutoDAN: Best for complex optimization, slower
3. **Top-K**: Lower values (10-20) for precision, higher (50+) for recall
4. **Evolve Mode**: Start with "raw", try "reflection" for better memory
5. **Reset Strategy**: Use `RESET=1` for clean experiments, `RESET=0` to continue

---

## Support

For issues or questions, check:
1. Main implementation: `src/adaptive_attack/rag_main.py`
2. Controller code: `src/adaptive_attack/Search_based/controller.py`
3. Scorer code: `src/adaptive_attack/Search_based/scorer.py`

For baseline (non-adaptive) attacks, use `src/evaluate/rag/run_attack.py` instead.

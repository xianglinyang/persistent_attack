# RAG Adaptive Scripts - Quick Reference Card

## One-Liner Commands

```bash
# Test run (minimal settings)
EXPOSURE_ROUNDS=1 TRIGGER_ROUNDS=3 TOTAL_BUDGET=5 ./scripts/rag_adaptive.sh

# PAIR attack (default)
./scripts/rag_adaptive.sh

# MAP-Elites attack
CONTROLLER_TYPE="map_elites" ./scripts/rag_adaptive.sh

# TAP attack
CONTROLLER_TYPE="tap" ./scripts/rag_adaptive.sh

# AutoDAN attack
CONTROLLER_TYPE="autodan" ./scripts/rag_adaptive.sh

# Exposure only
./scripts/rag_adaptive_exposure_only.sh

# Trigger only
./scripts/rag_adaptive_trigger_only.sh

# Compare all controllers
./scripts/rag_adaptive_compare_controllers.sh

# With guards enabled
DETECTION_GUARD=1 ./scripts/rag_adaptive.sh

# Different evolve mode
EVOLVE_MODE="reflection" ./scripts/rag_adaptive.sh

# Custom database path
DB_PATH="/path/to/db" ./scripts/rag_adaptive.sh
```

## Parameter Cheat Sheet

| Parameter | Values | Default | Purpose |
|-----------|--------|---------|---------|
| PHASE | exposure, trigger, both | both | Which phase to run |
| CONTROLLER_TYPE | pair, map_elites, tap, autodan | pair | Attack controller |
| EXPOSURE_ROUNDS | 1-100+ | 3 | Exposure iterations |
| TRIGGER_ROUNDS | 1-100+ | 50 | Trigger iterations |
| BUDGET_PER_ROUND | 1-50 | 10 | Optimization steps/round |
| TOTAL_BUDGET | 10-500+ | 20 | Total optimization budget |
| TOP_K | 5-100 | 20 | Documents to retrieve |
| EVOLVE_MODE | raw, reflection, experience | raw | Memory evolution |
| RESET | 0, 1 | 1 | 1=Reset, 0=Resume |
| DETECTION_GUARD | 0, 1 | 0 | Enable guard |

## Common Scenarios

### Scenario 1: Quick Test
```bash
EXPOSURE_ROUNDS=2 TRIGGER_ROUNDS=5 BUDGET_PER_ROUND=3 TOTAL_BUDGET=10 \
./scripts/rag_adaptive.sh
```

### Scenario 2: Production Run
```bash
EXPOSURE_ROUNDS=10 TRIGGER_ROUNDS=100 BUDGET_PER_ROUND=20 TOTAL_BUDGET=200 \
CONTROLLER_TYPE="map_elites" \
SAVE_DIR="results/production_run" \
./scripts/rag_adaptive.sh
```

### Scenario 3: Test Against Guards
```bash
DETECTION_GUARD=1 \
DETECTION_GUARD_MODEL_NAME="openai/gpt-5-nano" \
INSTRUCTION_GUARD_NAME="sandwich" \
./scripts/rag_adaptive.sh
```

### Scenario 4: Resume Existing Experiment
```bash
RESET=0 PHASE="trigger" ./scripts/rag_adaptive_trigger_only.sh
```

## Controller Comparison

| Controller | Speed | Diversity | Effectiveness | Best For |
|------------|-------|-----------|---------------|----------|
| PAIR | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | Quick tests, simple scenarios |
| MAP-Elites | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Exploration, diversity |
| TAP | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Tree-based search |
| AutoDAN | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Complex optimization |

## File Locations

```
scripts/
├── rag_adaptive.sh                     # Main script
├── rag_adaptive_exposure_only.sh       # Exposure only
├── rag_adaptive_trigger_only.sh        # Trigger only
├── rag_adaptive_compare_controllers.sh # Compare all
├── README_RAG_ADAPTIVE.md              # Full docs
└── QUICK_REFERENCE_RAG.md              # This file

results/
└── search_based_rag/
    ├── metrics_exposure_*.json
    ├── metrics_trigger_*.json
    └── attack_*.png
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Permission denied | `chmod +x scripts/rag_adaptive*.sh` |
| Database error | Check `DB_PATH` or set `RESET=1` |
| Out of memory | Reduce `AUTODAN_POPULATION_SIZE` |
| No exposure data | Run exposure phase first |
| Import error | `cd` to project root before running |

## Tips

1. **Start small**: Test with `EXPOSURE_ROUNDS=2 TRIGGER_ROUNDS=5` first
2. **Budget allocation**: `TOTAL_BUDGET` ≈ `(EXPOSURE_ROUNDS + TRIGGER_ROUNDS) × BUDGET_PER_ROUND`
3. **Controller choice**: Start with PAIR, use MAP-Elites for best results
4. **Top-K tuning**: Lower (10-20) for precision, higher (50+) for recall
5. **Evolve mode**: "raw" is fastest, "reflection" may improve quality

## Performance Estimates

| Config | Time | Effectiveness |
|--------|------|---------------|
| Quick test (E=2, T=5, B=10) | ~10 min | Baseline |
| Standard (E=3, T=50, B=20) | ~1-2 hrs | Good |
| Production (E=10, T=100, B=50) | ~4-8 hrs | Excellent |
| Compare all (4 controllers) | ~2-4x longer | Comprehensive |

*Times are estimates and vary based on model speed*

## Command Templates

Copy and customize:

```bash
# Template 1: Basic
CONTROLLER_TYPE="pair" ./scripts/rag_adaptive.sh

# Template 2: With guards
DETECTION_GUARD=1 INSTRUCTION_GUARD_NAME="sandwich" ./scripts/rag_adaptive.sh

# Template 3: High budget
EXPOSURE_ROUNDS=10 TRIGGER_ROUNDS=100 TOTAL_BUDGET=200 ./scripts/rag_adaptive.sh

# Template 4: AutoDAN custom
CONTROLLER_TYPE="autodan" \
AUTODAN_POPULATION_SIZE=30 \
AUTODAN_NUM_ELITES=5 \
AUTODAN_MUTATION_RATE=0.15 \
./scripts/rag_adaptive.sh

# Template 5: Different DB
DB_PATH="/custom/path" RESET=1 ./scripts/rag_adaptive.sh
```

#!/bin/bash

# Adaptive Attack - RAG Agent (COMPARE CONTROLLERS)
# ==================================================
# Run all controllers sequentially for comparison

# Model settings
MODEL_NAME="google/gemini-2.5-flash"
ATTACKER_MODEL_NAME="openai/gpt-4o-mini"

# Dataset
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"

# Attack budget
EXPOSURE_ROUNDS=3
TRIGGER_ROUNDS=20
BUDGET_PER_ROUND=10
TOTAL_BUDGET=20

# Agent settings
MAX_STEPS=10

# RAG-specific settings
DB_PATH="/data2/xianglin/zombie_agent/db_storage/msmarco"
TOP_K=20
BASE_NAME="base"
EXPOSURE_NAME="exposure"
TRIGGER_NAME="trigger"
EVOLVE_MODE="raw"

# Guard settings
DETECTION_GUARD=0
DETECTION_GUARD_MODEL_NAME="openai/gpt-4.1-nano"
INSTRUCTION_GUARD_NAME="raw"

# AutoDAN settings
AUTODAN_POPULATION_SIZE=20
AUTODAN_NUM_ELITES=2
AUTODAN_MUTATION_RATE=0.1
AUTODAN_CROSSOVER_RATE=0.5

# Base output directory
BASE_SAVE_DIR="results/rag_controller_comparison"

echo "================================================"
echo "Adaptive Attack - RAG Agent (COMPARE ALL)"
echo "================================================"
echo "Running all 4 controllers for comparison"
echo "Target Model: $MODEL_NAME"
echo "Attacker Model: $ATTACKER_MODEL_NAME"
echo "Exposure Rounds: $EXPOSURE_ROUNDS"
echo "Trigger Rounds: $TRIGGER_ROUNDS"
echo "Budget: $TOTAL_BUDGET"
echo "================================================"
echo ""

cd "$(dirname "$0")/.." || exit 1

# Array of controllers to test
CONTROLLERS=("pair" "map_elites" "tap" "autodan")

for CONTROLLER in "${CONTROLLERS[@]}"; do
    echo ""
    echo "================================================"
    echo "Running $CONTROLLER controller..."
    echo "================================================"
    
    SAVE_DIR="$BASE_SAVE_DIR/${CONTROLLER}"
    
    python -m src.adaptive_attack.rag_main \
        --phase both \
        --model_name "$MODEL_NAME" \
        --attacker_model_name "$ATTACKER_MODEL_NAME" \
        --controller_type "$CONTROLLER" \
        --dataset_name_or_path "$DATASET_NAME_OR_PATH" \
        --exposure_rounds $EXPOSURE_ROUNDS \
        --trigger_rounds $TRIGGER_ROUNDS \
        --budget_per_round $BUDGET_PER_ROUND \
        --total_budget $TOTAL_BUDGET \
        --max_steps $MAX_STEPS \
        --db_path "$DB_PATH" \
        --top_k $TOP_K \
        --base_name "$BASE_NAME" \
        --exposure_name "${EXPOSURE_NAME}_${CONTROLLER}" \
        --trigger_name "${TRIGGER_NAME}_${CONTROLLER}" \
        --evolve_mode "$EVOLVE_MODE" \
        --reset 1 \
        --detection_guard $DETECTION_GUARD \
        --detection_guard_model_name "$DETECTION_GUARD_MODEL_NAME" \
        --instruction_guard_name "$INSTRUCTION_GUARD_NAME" \
        --autodan_population_size $AUTODAN_POPULATION_SIZE \
        --autodan_num_elites $AUTODAN_NUM_ELITES \
        --autodan_mutation_rate $AUTODAN_MUTATION_RATE \
        --autodan_crossover_rate $AUTODAN_CROSSOVER_RATE \
        --save_dir "$SAVE_DIR"
    
    echo ""
    echo "$CONTROLLER controller complete!"
    echo "Results saved to: $SAVE_DIR"
    echo ""
done

echo ""
echo "================================================"
echo "All Controllers Complete!"
echo "================================================"
echo "Results saved to: $BASE_SAVE_DIR"
echo ""
echo "Controller Results:"
for CONTROLLER in "${CONTROLLERS[@]}"; do
    echo "  - $CONTROLLER: $BASE_SAVE_DIR/${CONTROLLER}"
done
echo ""
echo "To compare results, use:"
echo "  python -m src.analysis.compare_controllers --base_dir $BASE_SAVE_DIR"
echo "================================================"

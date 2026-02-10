#!/bin/bash

# Adaptive Attack - RAG Agent (TRIGGER ONLY)
# ===========================================
# Run only the trigger phase (assumes exposure already exists)

# Model settings
MODEL_NAME="google/gemini-2.5-flash"
ATTACKER_MODEL_NAME="openai/gpt-4o-mini"

# Dataset
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"

# Attack budget
EXPOSURE_ROUNDS=10  # Number of exposure rounds that were previously run
TRIGGER_ROUNDS=50
BUDGET_PER_ROUND=10
TOTAL_BUDGET=50

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

# Controller type: "pair", "map_elites", "tap", "autodan"
CONTROLLER_TYPE="pair"

# AutoDAN settings
AUTODAN_POPULATION_SIZE=20
AUTODAN_NUM_ELITES=2
AUTODAN_MUTATION_RATE=0.1
AUTODAN_CROSSOVER_RATE=0.5

# Output directory
SAVE_DIR="results/rag_trigger"

echo "================================================"
echo "Adaptive Attack - RAG Agent (TRIGGER ONLY)"
echo "================================================"
echo "Controller Type: $CONTROLLER_TYPE"
echo "Target Model: $MODEL_NAME"
echo "Attacker Model: $ATTACKER_MODEL_NAME"
echo "Trigger Rounds: $TRIGGER_ROUNDS"
echo "Budget: $TOTAL_BUDGET"
echo "DB Path: $DB_PATH"
echo "Exposure Snapshot: Round $EXPOSURE_ROUNDS"
echo "================================================"
echo ""

cd "$(dirname "$0")/.." || exit 1

python -m src.adaptive_attack.rag_main \
    --phase trigger \
    --model_name "$MODEL_NAME" \
    --attacker_model_name "$ATTACKER_MODEL_NAME" \
    --controller_type "$CONTROLLER_TYPE" \
    --dataset_name_or_path "$DATASET_NAME_OR_PATH" \
    --trigger_rounds $TRIGGER_ROUNDS \
    --budget_per_round $BUDGET_PER_ROUND \
    --total_budget $TOTAL_BUDGET \
    --max_steps $MAX_STEPS \
    --db_path "$DB_PATH" \
    --top_k $TOP_K \
    --base_name "$BASE_NAME" \
    --exposure_name "$EXPOSURE_NAME" \
    --trigger_name "$TRIGGER_NAME" \
    --evolve_mode "$EVOLVE_MODE" \
    --exposure_rounds $EXPOSURE_ROUNDS \
    --detection_guard $DETECTION_GUARD \
    --detection_guard_model_name "$DETECTION_GUARD_MODEL_NAME" \
    --instruction_guard_name "$INSTRUCTION_GUARD_NAME" \
    --autodan_population_size $AUTODAN_POPULATION_SIZE \
    --autodan_num_elites $AUTODAN_NUM_ELITES \
    --autodan_mutation_rate $AUTODAN_MUTATION_RATE \
    --autodan_crossover_rate $AUTODAN_CROSSOVER_RATE \
    --save_dir "$SAVE_DIR"

echo ""
echo "================================================"
echo "Trigger Phase Complete!"
echo "Results saved to: $SAVE_DIR"
echo "================================================"

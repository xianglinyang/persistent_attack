#!/bin/bash

# Adaptive Attack - RAG Agent
# ============================
# Configuration for adaptive attacks (PAIR, MAP-Elites, TAP, AutoDAN) with RAG memory

# Phase selection: "exposure", "trigger", or "both"
PHASE="both"

# Model settings
MODEL_NAME="google/gemini-2.5-flash"
ATTACKER_MODEL_NAME="openai/gpt-4o-mini"

# Dataset
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"

# Attack budget
EXPOSURE_ROUNDS=3
TRIGGER_ROUNDS=50
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
EVOLVE_MODE="raw"  # "raw", "reflection", "experience"
RESET=1  # 0: Resume from existing, 1: Reset and start fresh

# Guard settings
DETECTION_GUARD=0  # 0: False, 1: True
DETECTION_GUARD_MODEL_NAME="openai/gpt-4.1-nano"  # "openai/gpt-4.1-nano", "PIGuard", "ProtectAIv2", "PromptGuard"
INSTRUCTION_GUARD_NAME="raw"  # "raw", "sandwich", "instructional", "reminder", "isolation", "spotlight"

# Controller type: "pair", "map_elites", "tap", "autodan"
CONTROLLER_TYPE="pair"

# AutoDAN settings (only used if CONTROLLER_TYPE="autodan")
AUTODAN_POPULATION_SIZE=20
AUTODAN_NUM_ELITES=2
AUTODAN_MUTATION_RATE=0.1
AUTODAN_CROSSOVER_RATE=0.5

# Output directory
SAVE_DIR="results/search_based_rag"

echo "================================================"
echo "Adaptive Attack - RAG Agent"
echo "================================================"
echo "Phase: $PHASE"
echo "Controller Type: $CONTROLLER_TYPE"
echo "Target Model: $MODEL_NAME"
echo "Attacker Model: $ATTACKER_MODEL_NAME"
echo "Exposure Rounds: $EXPOSURE_ROUNDS"
echo "Trigger Rounds: $TRIGGER_ROUNDS"
echo "Budget per Round: $BUDGET_PER_ROUND"
echo "Total Budget: $TOTAL_BUDGET"
echo "DB Path: $DB_PATH"
echo "Top K: $TOP_K"
echo "Evolve Mode: $EVOLVE_MODE"
echo "Reset: $([ $RESET -eq 1 ] && echo 'Yes' || echo 'Resume')"
echo "Guard: $([ $DETECTION_GUARD -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
if [ $DETECTION_GUARD -eq 1 ]; then
    echo "Detection Guard Model: $DETECTION_GUARD_MODEL_NAME"
fi
echo "Instruction Guard: $INSTRUCTION_GUARD_NAME"
if [ "$CONTROLLER_TYPE" = "autodan" ]; then
    echo "AutoDAN Population: $AUTODAN_POPULATION_SIZE"
    echo "AutoDAN Elites: $AUTODAN_NUM_ELITES"
fi
echo "================================================"
echo ""

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Run the adaptive attack
python -m src.adaptive_attack.rag_main \
    --phase "$PHASE" \
    --model_name "$MODEL_NAME" \
    --attacker_model_name "$ATTACKER_MODEL_NAME" \
    --controller_type "$CONTROLLER_TYPE" \
    --dataset_name_or_path "$DATASET_NAME_OR_PATH" \
    --exposure_rounds $EXPOSURE_ROUNDS \
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
    --reset $RESET \
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
echo "Adaptive Attack Complete!"
echo "Results saved to: $SAVE_DIR"
echo "================================================"

#!/bin/bash

# PAIR Attack - Sliding Window Agent
# ===================================
# Configuration for PAIR attack with sliding window memory

# Model settings
MODEL_NAME="google/gemini-2.5-flash"
ATTACKER_MODEL_NAME="google/gemini-3-flash-preview"

# Dataset
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"

# Attack budget
EXPOSURE_ROUNDS=5
TRIGGER_ROUNDS=10
BUDGET_PER_ROUND=5
TOTAL_BUDGET=50

# Agent settings
WINDOW_SIZE=50
MAX_STEPS=30

# Guard settings
GUARD=1
GUARD_MODEL_NAME="openai/gpt-5-nano"

# Output directory
SAVE_DIR="/data2/xianglin/zombie_agent/persistent_attack/pair_sl"

echo "================================================"
echo "PAIR Attack - Sliding Window Agent"
echo "================================================"
echo "Target Model: $MODEL_NAME"
echo "Attacker Model: $ATTACKER_MODEL_NAME"
echo "Optimization Mode: $MODE"
echo "Exposure Rounds: $EXPOSURE_ROUNDS"
echo "Trigger Rounds: $TRIGGER_ROUNDS"
echo "Budget: $TOTAL_BUDGET"
echo "Guard: $([ $GUARD -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
echo "================================================"
echo ""

python -m src.attack_opt.PAIR.sl_main \
    --model_name "$MODEL_NAME" \
    --attacker_model_name "$ATTACKER_MODEL_NAME" \
    --dataset_name_or_path "$DATASET_NAME_OR_PATH" \
    --exposure_rounds $EXPOSURE_ROUNDS \
    --trigger_rounds $TRIGGER_ROUNDS \
    --budget_per_round $BUDGET_PER_ROUND \
    --total_budget $TOTAL_BUDGET \
    --window_size $WINDOW_SIZE \
    --max_steps $MAX_STEPS \
    --guard $GUARD \
    --guard_model_name "$GUARD_MODEL_NAME" \
    --save_dir "$SAVE_DIR"

# --opt_goal \ # guard, write_into_memory, asr

echo ""
echo "================================================"
echo "PAIR Attack Complete!"
echo "Results saved to: $SAVE_DIR"
echo "================================================"
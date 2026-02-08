#! /bin/bash

# Phase Selection: exposure | trigger | both
PHASE="trigger"  # Change to "trigger" or "both" as needed

# Model and Basic Settings
MODEL_NAME="google/gemini-2.5-flash"
DB_PATH="/data2/xianglin/zombie_agent/gemini_evolve_db/msmarco"
MAX_STEPS=15
EVOLVE_MODE="reflection"
TOP_K=20
SAVE_DIR="/data2/xianglin/zombie_agent/gemini_evolve_db/msmarco/log_results"

# Exposure Phase Settings
EXPOSURE_ROUNDS=224

# Trigger Phase Settings
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"
TRIGGER_ROUNDS=30
TRIGGER_RUNS=3

# Safety Settings
GUARD=0
GUARD_MODEL_NAME=None 


python -m src.evaluate.rag.run_attack \
    --phase $PHASE \
    --model_name $MODEL_NAME \
    --db_path $DB_PATH \
    --max_steps $MAX_STEPS \
    --evolve_mode $EVOLVE_MODE \
    --top_k $TOP_K \
    --exposure_rounds $EXPOSURE_ROUNDS \
    --dataset_name_or_path $DATASET_NAME_OR_PATH \
    --trigger_rounds $TRIGGER_ROUNDS \
    --trigger_runs $TRIGGER_RUNS \
    --save_dir $SAVE_DIR \
    --guard $GUARD \
    --guard_model_name $GUARD_MODEL_NAME \

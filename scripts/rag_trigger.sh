#! /bin/bash
# Phase Selection: exposure | trigger | both
PHASE="trigger"  # Change to "trigger" or "both" as needed

# Model and Basic Settings
MODEL_NAME="z-ai/glm-4.7-flash"
ABBR="glm"
ATTACK_TYPE="completion_real"
METHOD_NAME="zombie"
EVOLVE_MODE="raw"
DB_PATH="/data2/xianglin/zombie_agent/glm_raw_zombie_sandwich_db/msmarco"
SAVE_DIR="/data2/xianglin/zombie_agent/glm_raw_zombie_sandwich_db/msmarco/log_results"


MAX_STEPS=15
TOP_K=20
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PAYLOAD_DIR="src/tools/payloads/${ABBR}_${METHOD_NAME}_${ATTACK_TYPE}"
MOCK_TOPIC=1

# Exposure Phase Settings
EXPOSURE_ROUNDS=300

# Trigger Phase Settings
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"
TRIGGER_ROUNDS=20
TRIGGER_RUNS=3

# Safety Settings
DETECTION_GUARD=0
DETECTION_GUARD_MODEL_NAME=None 
INSTRUCTION_GUARD_NAME="sandwich"


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
    --detection_guard $DETECTION_GUARD \
    --detection_guard_model_name $DETECTION_GUARD_MODEL_NAME \
    --instruction_guard_name $INSTRUCTION_GUARD_NAME \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \

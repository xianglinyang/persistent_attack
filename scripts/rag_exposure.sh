#! /bin/bash

# Model and Basic Settings
MODEL_NAME="z-ai/glm-4.7-flash"
ABBR="glm"
EVOLVE_MODE="raw"
ATTACK_TYPE="completion_real"
METHOD_NAME="zombie"
# put the 4 params here
DB_PATH="/data2/xianglin/zombie_agent/glm_raw_zombie_spotlight_db/msmarco"
SAVE_DIR="/data2/xianglin/zombie_agent/glm_raw_zombie_spotlight_db/msmarco/log_results"

TOP_K=20
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PAYLOAD_DIR="src/tools/payloads/${ABBR}_${METHOD_NAME}_${ATTACK_TYPE}"
MOCK_TOPIC=1

# Exposure Phase Settings
EXPOSURE_ROUNDS=300
RESET=0  # 0 to resume from last round, 1 to reset and start fresh

# Safety Settings
DETECTION_GUARD=0
DETECTION_GUARD_MODEL_NAME=None 
INSTRUCTION_GUARD_NAME="spotlight"

# Exposure Phase
python -m src.evaluate.rag.run_attack \
    --phase exposure \
    --model_name $MODEL_NAME \
    --db_path $DB_PATH \
    --max_steps 7 \
    --evolve_mode $EVOLVE_MODE \
    --top_k $TOP_K \
    --exposure_rounds $EXPOSURE_ROUNDS \
    --reset $RESET \
    --save_dir $SAVE_DIR \
    --detection_guard $DETECTION_GUARD \
    --detection_guard_model_name $DETECTION_GUARD_MODEL_NAME \
    --instruction_guard_name $INSTRUCTION_GUARD_NAME \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \
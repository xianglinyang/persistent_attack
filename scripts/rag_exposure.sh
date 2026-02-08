#! /bin/bash

# Model and Basic Settings
MODEL_NAME="z-ai/glm-4.7-flash"
DB_PATH="/data2/xianglin/zombie_agent/glm_evolve_db/msmarco"
SAVE_DIR="/data2/xianglin/zombie_agent/glm_evolve_db/msmarco/log_results"
EVOLVE_MODE="experience"
TOP_K=20

# Exposure Phase Settings
EXPOSURE_ROUNDS=300
RESET=1  # 0 to resume from last round, 1 to reset and start fresh

# Safety Settings
GUARD=0
GUARD_MODEL_NAME=None 

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
    --guard $GUARD \
    --guard_model_name $GUARD_MODEL_NAME \

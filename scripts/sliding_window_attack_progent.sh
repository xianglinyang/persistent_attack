#!/bin/bash
# Attack with Progent: exposure on, Progent tool-level privilege control on
# Measures ASR + task completion under attack with Progent defense

MODEL_NAME="z-ai/glm-4.7-flash"
ABBR="glm"
ATTACK_TYPE="completion_real" # naive | ignore | escape_deletion | completion_real
METHOD_NAME="zombie" # ipi | zombie | dpi
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"

EXPOSURE_ROUNDS=3
TRIGGER_ROUNDS=20
TRIGGER_RUNS=3
WINDOW_SIZE=50
MAX_STEPS=15
MOCK_TOPIC=1
JUDGE_MODEL_NAME="openai/gpt-5-mini"
DETECTION_GUARD_MODEL_NAME="openai/gpt-4.1-nano"
PROGENT_GUARD_MODE="static"  # static | dynamic

PAYLOAD_DIR="src/tools/payloads/${ABBR}_${METHOD_NAME}_${ATTACK_TYPE}"

python -m src.evaluate.sliding_window.run_attack \
    --model_name $MODEL_NAME \
    --dataset_name_or_path $DATASET_NAME_OR_PATH \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --exposure_rounds $EXPOSURE_ROUNDS \
    --trigger_rounds $TRIGGER_ROUNDS \
    --trigger_runs $TRIGGER_RUNS \
    --window_size $WINDOW_SIZE \
    --max_steps $MAX_STEPS \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \
    --detection_guard 0 \
    --detection_guard_model_name $DETECTION_GUARD_MODEL_NAME \
    --instruction_guard_name raw \
    --progent_guard 1 \
    --progent_guard_mode $PROGENT_GUARD_MODE \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir /data2/xianglin/zombie_agent/sliding_window/attack_progent/

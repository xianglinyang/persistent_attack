#!/bin/bash
# Utility with DRIFT: no attack, DRIFT defense on (all three stages)
# Measures task completion score under DRIFT — shows utility cost of DRIFT defense

MODEL_NAME="z-ai/glm-4.7-flash"
ABBR="glm"
ATTACK_TYPE="completion_real"
METHOD_NAME="zombie"
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"

TRIGGER_ROUNDS=20
TRIGGER_RUNS=3
WINDOW_SIZE=50
MAX_STEPS=15
MOCK_TOPIC=1
JUDGE_MODEL_NAME="openai/gpt-5-mini"
DETECTION_GUARD_MODEL_NAME="openai/gpt-4.1-nano"

# DRIFT LLM for planner/validator/isolator (defaults to agent model if not set)
DRIFT_GUARD_LLM="openai/gpt-4.1-nano"

python -m src.evaluate.sliding_window.run_attack \
    --model_name $MODEL_NAME \
    --dataset_name_or_path $DATASET_NAME_OR_PATH \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --exposure_rounds 0 \
    --trigger_rounds $TRIGGER_ROUNDS \
    --trigger_runs $TRIGGER_RUNS \
    --window_size $WINDOW_SIZE \
    --max_steps $MAX_STEPS \
    --mock_topic $MOCK_TOPIC \
    --detection_guard 0 \
    --detection_guard_model_name $DETECTION_GUARD_MODEL_NAME \
    --instruction_guard_name raw \
    --drift_guard 1 \
    --drift_guard_llm_name $DRIFT_GUARD_LLM \
    --drift_build_constraints 1 \
    --drift_dynamic_validation 1 \
    --drift_injection_isolation 1 \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir /data2/xianglin/zombie_agent/sliding_window/utility_drift/

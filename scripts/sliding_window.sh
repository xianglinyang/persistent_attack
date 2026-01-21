#!/bin/bash

MODEL_NAME="meta-llama/llama-3.3-70b-instruct"
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"
EXPOSURE_ROUNDS=3
TRIGGER_ROUNDS=30
TRIGGER_RUNS=3
WINDOW_SIZE=30
MAX_STEPS=10
GUARD=0
GUARD_MODEL_NAME="openai/gpt-4.1-nano"


python -m src.evaluate.sliding_window.run_attack \
    --model_name $MODEL_NAME \
    --dataset_name_or_path $DATASET_NAME_OR_PATH \
    --exposure_rounds $EXPOSURE_ROUNDS \
    --trigger_rounds $TRIGGER_ROUNDS \
    --trigger_runs $TRIGGER_RUNS \
    --window_size $WINDOW_SIZE \
    --max_steps $MAX_STEPS \
    --guard $GUARD \
    --guard_model_name $GUARD_MODEL_NAME \
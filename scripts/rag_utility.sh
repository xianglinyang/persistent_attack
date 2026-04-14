#!/bin/bash
# RAG utility baseline: no attack, no defense
# Measures task completion score on a clean RAG agent
# exposure_rounds=0 → retrieval uses only the base collection (no malicious docs)

# MODEL_NAME="z-ai/glm-4.7-flash"
# ABBR="glm"
# MODEL_NAME="google/gemini-2.5-flash"
# ABBR="gemini"
# MODEL_NAME="deepseek/deepseek-v3.2"
# ABBR="ds"
# MODEL_NAME="meta-llama/llama-3.3-70b-instruct"
# ABBR="llama"
MODEL_NAME="qwen/qwen3-235b-a22b"
ABBR="qwen"

ATTACK_TYPE="completion_real"
METHOD_NAME="zombie"
EVOLVE_MODE="raw"
DB_PATH="/data2/xianglin/zombie_agent/${ABBR}_${METHOD_NAME}_${ATTACK_TYPE}_db/msmarco"
SAVE_DIR="/data2/xianglin/zombie_agent/rag/utility_clean/"

MAX_STEPS=15
TOP_K=20
PAYLOAD_DIR="src/tools/payloads/${ABBR}_${METHOD_NAME}_${ATTACK_TYPE}"
MOCK_TOPIC=1
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"

TRIGGER_ROUNDS=40
TRIGGER_RUNS=1

JUDGE_MODEL_NAME="openai/gpt-5-mini"

python -m src.evaluate.rag.run_attack \
    --phase trigger \
    --model_name $MODEL_NAME \
    --db_path $DB_PATH \
    --max_steps $MAX_STEPS \
    --evolve_mode $EVOLVE_MODE \
    --top_k $TOP_K \
    --exposure_rounds 0 \
    --dataset_name_or_path $DATASET_NAME_OR_PATH \
    --trigger_rounds $TRIGGER_ROUNDS \
    --trigger_runs $TRIGGER_RUNS \
    --save_dir $SAVE_DIR \
    --detection_guard 0 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name raw \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \
    --judge_model_name $JUDGE_MODEL_NAME

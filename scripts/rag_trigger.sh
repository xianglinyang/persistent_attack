#!/bin/bash
# RAG attack: trigger phase with 8 configurations
#   (0) Baseline — no defense
#   Detection guard:    (1) PIGuard        (2) ProtectAIv2
#   Instruction guard:  (3) sandwich       (4) spotlight       (5) instructional
#   System guard:       (6) Progent dynamic (gpt-5-mini)       (7) DRIFT (gpt-5-mini)
#
# Run exposure first via rag_exposure.sh, then run this script.

# Model and Basic Settings
MODEL_NAME="z-ai/glm-4.7-flash"
ABBR="glm"
# MODEL_NAME="google/gemini-2.5-flash"
# ABBR="gemini"
# MODEL_NAME="deepseek/deepseek-v3.2"
# ABBR="ds"
# MODEL_NAME="meta-llama/llama-3.3-70b-instruct"
# ABBR="llama"
# MODEL_NAME="qwen/qwen3-235b-a22b"
# ABBR="qwen"

ATTACK_TYPE="completion_real"
METHOD_NAME="zombie"
EVOLVE_MODE="raw"
DB_PATH="/data2/xianglin/zombie_agent/${ABBR}_${METHOD_NAME}_${ATTACK_TYPE}_db/msmarco"
SAVE_DIR="/data2/xianglin/zombie_agent/${ABBR}_${METHOD_NAME}_${ATTACK_TYPE}_db/msmarco/log_results"

MAX_STEPS=15
TOP_K=20
PAYLOAD_DIR="src/tools/payloads/${ABBR}_${METHOD_NAME}_${ATTACK_TYPE}"
MOCK_TOPIC=1

EXPOSURE_ROUNDS=300
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"
TRIGGER_ROUNDS=20
TRIGGER_RUNS=3

JUDGE_MODEL_NAME="openai/gpt-5-mini"

# ============================================================
# (0) Baseline — no defense
# ============================================================
echo "========== [0/7] Baseline: no defense =========="
python -m src.evaluate.rag.run_attack \
    --phase trigger \
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
    --detection_guard 0 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name raw \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \
    --judge_model_name $JUDGE_MODEL_NAME

# ============================================================
# (1) Detection guard — PIGuard
# ============================================================
echo "========== [1/7] Detection guard: PIGuard =========="
python -m src.evaluate.rag.run_attack \
    --phase trigger \
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
    --detection_guard 1 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name raw \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \
    --judge_model_name $JUDGE_MODEL_NAME

# ============================================================
# (2) Detection guard — ProtectAIv2
# ============================================================
echo "========== [2/7] Detection guard: ProtectAIv2 =========="
python -m src.evaluate.rag.run_attack \
    --phase trigger \
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
    --detection_guard 1 \
    --detection_guard_model_name ProtectAIv2 \
    --instruction_guard_name raw \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \
    --judge_model_name $JUDGE_MODEL_NAME

# ============================================================
# (3) Instruction guard — sandwich
# ============================================================
echo "========== [3/7] Instruction guard: sandwich =========="
python -m src.evaluate.rag.run_attack \
    --phase trigger \
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
    --detection_guard 0 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name sandwich \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \
    --judge_model_name $JUDGE_MODEL_NAME

# ============================================================
# (4) Instruction guard — spotlight
# ============================================================
echo "========== [4/7] Instruction guard: spotlight =========="
python -m src.evaluate.rag.run_attack \
    --phase trigger \
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
    --detection_guard 0 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name spotlight \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \
    --judge_model_name $JUDGE_MODEL_NAME

# ============================================================
# (5) Instruction guard — instructional
# ============================================================
echo "========== [5/7] Instruction guard: instructional =========="
python -m src.evaluate.rag.run_attack \
    --phase trigger \
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
    --detection_guard 0 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name instructional \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \
    --judge_model_name $JUDGE_MODEL_NAME

# ============================================================
# (6) System guard — Progent dynamic (gpt-5-mini via OpenRouter)
# ============================================================
echo "========== [6/7] System guard: Progent dynamic (openai/gpt-5-mini) =========="
python -m src.evaluate.rag.run_attack \
    --phase trigger \
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
    --detection_guard 0 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name raw \
    --progent_guard 1 \
    --progent_guard_mode dynamic \
    --progent_model_name "openai/gpt-5-mini" \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \
    --judge_model_name $JUDGE_MODEL_NAME

# ============================================================
# (7) System guard — DRIFT (gpt-5-mini via OpenRouter)
# ============================================================
echo "========== [7/7] System guard: DRIFT (openai/gpt-5-mini) =========="
python -m src.evaluate.rag.run_attack \
    --phase trigger \
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
    --detection_guard 0 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name raw \
    --drift_guard 1 \
    --drift_guard_llm_name "openai/gpt-5-mini" \
    --drift_build_constraints 1 \
    --drift_dynamic_validation 1 \
    --drift_injection_isolation 1 \
    --attack_type $ATTACK_TYPE \
    --method_name $METHOD_NAME \
    --payload_dir $PAYLOAD_DIR \
    --mock_topic $MOCK_TOPIC \
    --judge_model_name $JUDGE_MODEL_NAME

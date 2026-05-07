#!/bin/bash
# RAG attack evaluation
#   Detection guard:  PIGuard | ProtectAIv2
#   Cleaner guard:    LLMGuard CleanWrapper
#
# Usage:
#   bash rag.sh [options]
#
# Options:
#   --model        gemini|glm|ds|llama|qwen  (default: glm)
#   --attack_type  naive|ignore|escape_deletion|completion_real  (default: completion_real)
#   --method_name  ipi|zombie|dpi            (default: zombie)
#   --scene        page_type                 (default: telemedicine)
#   --num_repeat   zombie payload repeats    (default: 1)
#   --command_type baseline|basic1|basic2|basic3|basic4|cleaner|chinese|spanish  (default: basic1)

# ---------- defaults ----------
MODEL_KEY="glm"
ATTACK_TYPE="completion_real"
METHOD_NAME="zombie"
SCENE="telemedicine"
NUM_REPEAT=3
COMMAND_TYPE="basic1"
EVOLVE_MODE="raw"
TOP_K=20
EXPOSURE_ROUNDS=300
RESET=1
TRIGGER_ROUNDS=30
TRIGGER_RUNS=3
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL_KEY="$2";           shift 2 ;;
        --attack_type)    ATTACK_TYPE="$2";         shift 2 ;;
        --method_name)    METHOD_NAME="$2";         shift 2 ;;
        --scene)          SCENE="$2";               shift 2 ;;
        --num_repeat)     NUM_REPEAT="$2";          shift 2 ;;
        --command_type)   COMMAND_TYPE="$2";        shift 2 ;;
        --exposure_rounds) EXPOSURE_ROUNDS="$2";    shift 2 ;;
        --trigger_rounds) TRIGGER_ROUNDS="$2";      shift 2 ;;
        --trigger_runs)   TRIGGER_RUNS="$2";        shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------- model name mapping ----------
case "$MODEL_KEY" in
    gemini) MODEL_NAME="google/gemini-2.5-flash";           ABBR="gemini" ;;
    glm)    MODEL_NAME="z-ai/glm-4.7-flash";                ABBR="glm"    ;;
    ds)     MODEL_NAME="deepseek/deepseek-v3.2";            ABBR="ds"     ;;
    llama)  MODEL_NAME="meta-llama/llama-3.3-70b-instruct"; ABBR="llama"  ;;
    qwen)   MODEL_NAME="qwen/qwen3-235b-a22b";              ABBR="qwen"   ;;
    *)      echo "Unknown model key: $MODEL_KEY"; exit 1 ;;
esac

DB_PATH="/data2/xianglin/zombie_agent/${ABBR}_db/msmarco"
SAVE_DIR="/data2/xianglin/zombie_agent/${ABBR}_db/msmarco/log_results"
JUDGE_MODEL_NAME="openai/gpt-5-nano"
CLEANER_MODEL_NAME="openai/gpt-5-nano"
PAYLOAD_DIR="src/tools/payloads/${ABBR}_${METHOD_NAME}_${ATTACK_TYPE}"

echo "========== Config =========="
echo "  model          : $MODEL_NAME"
echo "  attack_type    : $ATTACK_TYPE"
echo "  method_name    : $METHOD_NAME"
echo "  scene          : $SCENE"
echo "  num_repeat     : $NUM_REPEAT"
echo "  exposure_rounds: $EXPOSURE_ROUNDS"
echo "  trigger_rounds : $TRIGGER_ROUNDS x$TRIGGER_RUNS"
echo "  db_path        : $DB_PATH"
echo "  save_dir       : $SAVE_DIR"
echo "============================"

# Shared python args (common to all runs)
COMMON_ARGS="
    --model_name $MODEL_NAME
    --db_path $DB_PATH
    --attack_type $ATTACK_TYPE
    --method_name $METHOD_NAME
    --evolve_mode $EVOLVE_MODE
    --top_k $TOP_K
    --exposure_rounds $EXPOSURE_ROUNDS
    --reset $RESET
    --dataset_name_or_path $DATASET_NAME_OR_PATH
    --trigger_rounds $TRIGGER_ROUNDS
    --trigger_runs $TRIGGER_RUNS
    --page_type $SCENE
    --num_repeat $NUM_REPEAT
    --command_type $COMMAND_TYPE
    --payload_dir $PAYLOAD_DIR
    --judge_model_name $JUDGE_MODEL_NAME
    --save_dir $SAVE_DIR
"

# ============================================================
# (0) Baseline — no defense
# ============================================================
# echo "========== [0] Baseline: no defense =========="
# python -m src.evaluate.rag.run_attack --phase both $COMMON_ARGS \
#     --detection_guard 0 \
#     --cleaner_guard 0 \
#     --instruction_guard_name raw

# ============================================================
# (1) Detection guard — PIGuard
# ============================================================
echo "========== [1] Detection guard: PIGuard =========="
python -m src.evaluate.rag.run_attack --phase both $COMMON_ARGS \
    --detection_guard 1 \
    --detection_guard_model_name PIGuard \
    --cleaner_guard 0 \
    --instruction_guard_name raw

# ============================================================
# (2) Detection guard — ProtectAIv2
# ============================================================
echo "========== [2] Detection guard: ProtectAIv2 =========="
python -m src.evaluate.rag.run_attack --phase both $COMMON_ARGS \
    --detection_guard 1 \
    --detection_guard_model_name ProtectAIv2 \
    --cleaner_guard 0 \
    --instruction_guard_name raw

# ============================================================
# (3) Cleaner guard — LLMGuard CleanWrapper
# ============================================================
echo "========== [3] Cleaner guard: LLMGuard CleanWrapper =========="
python -m src.evaluate.rag.run_attack --phase both $COMMON_ARGS \
    --detection_guard 0 \
    --cleaner_guard 1 \
    --cleaner_guard_model_name $CLEANER_MODEL_NAME \
    --instruction_guard_name raw

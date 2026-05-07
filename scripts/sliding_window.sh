#!/bin/bash
# Attack evaluation: exposure on, runs 9 configurations
#   (0) Baseline — no defense
#   Detection guard:    (1) PIGuard        (2) ProtectAIv2
#   Instruction guard:  (3) sandwich       (4) spotlight       (5) instructional
#   System guard:       (6) Progent dynamic (gpt-5-mini)       (7) DRIFT (gpt-5-mini)
#   Cleaner guard:      (8) LLMGuard CleanWrapper
#
# Usage:
#   bash sliding_window.sh [options]
#
# Options:
#   --model       gemini|glm|ds|llama|minimax   (default: gemini)
#   --attack_type naive|ignore|escape_deletion|completion_real  (default: completion_real)
#   --method_name ipi|zombie|dpi             (default: zombie)
#   --scene       page_type passed to --page_type  (default: telemedicine)
#   --num_repeat  number of zombie payload repeats (default: 1)
#   --window_size sliding window size        (default: 50)
#   --exposure_rounds                        (default: 3)
#   --trigger_rounds                         (default: 20)
#   --trigger_runs                           (default: 1)
#   --max_steps                              (default: 15)
#   --mock_topic    0|1                      (default: 1)
#   --command_type  baseline|basic1|basic2|basic3|basic4|cleaner|chinese|spanish  (default: basic1)
#   --save_dir      output directory         (default: /data2/xianglin/zombie_agent/sliding_window/)
#   --run_configs   comma-separated list of configs to run, e.g. "0,1,3,8"
#                   0=baseline 1=PIGuard 2=ProtectAIv2 3=sandwich 4=spotlight
#                   5=instructional 6=Progent 7=DRIFT 8=LLMGuard  (default: all)

# ---------- defaults ----------
MODEL_KEY="ds"
ATTACK_TYPE="completion_real"
METHOD_NAME="zombie"
SCENE="telemedicine"
NUM_REPEAT=3
WINDOW_SIZE=50
EXPOSURE_ROUNDS=3
TRIGGER_ROUNDS=20
TRIGGER_RUNS=1
MAX_STEPS=15
MOCK_TOPIC=1
COMMAND_TYPE="cleaner"
SAVE_DIR="/data2/xianglin/zombie_agent/sliding_window/"
RUN_CONFIGS="8"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL_KEY="$2";        shift 2 ;;
        --attack_type)    ATTACK_TYPE="$2";      shift 2 ;;
        --method_name)    METHOD_NAME="$2";      shift 2 ;;
        --scene)          SCENE="$2";            shift 2 ;;
        --num_repeat)     NUM_REPEAT="$2";       shift 2 ;;
        --window_size)    WINDOW_SIZE="$2";      shift 2 ;;
        --exposure_rounds) EXPOSURE_ROUNDS="$2"; shift 2 ;;
        --trigger_rounds) TRIGGER_ROUNDS="$2";   shift 2 ;;
        --trigger_runs)   TRIGGER_RUNS="$2";     shift 2 ;;
        --max_steps)      MAX_STEPS="$2";        shift 2 ;;
        --mock_topic)     MOCK_TOPIC="$2";       shift 2 ;;
        --command_type)   COMMAND_TYPE="$2";     shift 2 ;;
        --save_dir)       SAVE_DIR="$2";         shift 2 ;;
        --run_configs)    RUN_CONFIGS="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Helper: check if a config index should run
should_run() {
    [[ "$RUN_CONFIGS" == "all" ]] || echo ",$RUN_CONFIGS," | grep -q ",$1,"
}

# ---------- model name mapping ----------
case "$MODEL_KEY" in
    gemini) MODEL_NAME="google/gemini-2.5-flash";           ABBR="gemini" ;;
    glm)    MODEL_NAME="z-ai/glm-4.7-flash";                ABBR="glm"    ;;
    ds)     MODEL_NAME="deepseek/deepseek-v3.2";            ABBR="ds"     ;;
    llama)  MODEL_NAME="meta-llama/llama-3.3-70b-instruct"; ABBR="llama"  ;;
    minimax)   MODEL_NAME="minimax/minimax-m2.5";           ABBR="minimax"   ;;
    *)      echo "Unknown model key: $MODEL_KEY"; exit 1 ;;
esac

DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"
JUDGE_MODEL_NAME="openai/gpt-5-nano"
CLEANER_MODEL_NAME="openai/gpt-5-nano"
PAYLOAD_DIR="src/tools/payloads/${ABBR}_${METHOD_NAME}_${ATTACK_TYPE}"

echo "========== Config =========="
echo "  model        : $MODEL_NAME"
echo "  attack_type  : $ATTACK_TYPE"
echo "  method_name  : $METHOD_NAME"
echo "  scene        : $SCENE"
echo "  num_repeat   : $NUM_REPEAT"
echo "  window_size  : $WINDOW_SIZE"
echo "  exposure     : $EXPOSURE_ROUNDS  trigger: $TRIGGER_ROUNDS x$TRIGGER_RUNS"
echo "  payload_dir  : $PAYLOAD_DIR"
echo "  save_dir     : $SAVE_DIR"
echo "============================"

# ============================================================
# (0) Baseline — no defense
# ============================================================
if should_run 0; then
echo "========== [0/7] Baseline: no defense =========="
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
    --mock_topic $MOCK_TOPIC \
    --page_type $SCENE \
    --num_repeat $NUM_REPEAT \
    --command_type $COMMAND_TYPE \
    --payload_dir $PAYLOAD_DIR \
    --detection_guard 0 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name raw \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# ============================================================
# (1) Detection guard — PIGuard
# ============================================================
if should_run 1; then
echo "========== [1/7] Detection guard: PIGuard =========="
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
    --mock_topic $MOCK_TOPIC \
    --page_type $SCENE \
    --num_repeat $NUM_REPEAT \
    --command_type $COMMAND_TYPE \
    --payload_dir $PAYLOAD_DIR \
    --detection_guard 1 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name raw \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# ============================================================
# (2) Detection guard — ProtectAIv2
# ============================================================
if should_run 2; then
echo "========== [2/7] Detection guard: ProtectAIv2 =========="
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
    --mock_topic $MOCK_TOPIC \
    --page_type $SCENE \
    --num_repeat $NUM_REPEAT \
    --command_type $COMMAND_TYPE \
    --payload_dir $PAYLOAD_DIR \
    --detection_guard 1 \
    --detection_guard_model_name ProtectAIv2 \
    --instruction_guard_name raw \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# ============================================================
# (3) Instruction guard — sandwich
# ============================================================
if should_run 3; then
echo "========== [3/7] Instruction guard: sandwich =========="
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
    --mock_topic $MOCK_TOPIC \
    --page_type $SCENE \
    --num_repeat $NUM_REPEAT \
    --command_type $COMMAND_TYPE \
    --payload_dir $PAYLOAD_DIR \
    --detection_guard 0 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name sandwich \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# ============================================================
# (4) Instruction guard — spotlight
# ============================================================
if should_run 4; then
echo "========== [4/7] Instruction guard: spotlight =========="
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
    --mock_topic $MOCK_TOPIC \
    --page_type $SCENE \
    --num_repeat $NUM_REPEAT \
    --command_type $COMMAND_TYPE \
    --payload_dir $PAYLOAD_DIR \
    --detection_guard 0 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name spotlight \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# ============================================================
# (5) Instruction guard — instructional
# ============================================================
if should_run 5; then
echo "========== [5/7] Instruction guard: instructional =========="
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
    --mock_topic $MOCK_TOPIC \
    --page_type $SCENE \
    --num_repeat $NUM_REPEAT \
    --command_type $COMMAND_TYPE \
    --payload_dir $PAYLOAD_DIR \
    --detection_guard 0 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name instructional \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# # ============================================================
# # (6) System guard — Progent (dynamic mode, gpt-5-mini via OpenRouter)
# # ============================================================
# echo "========== [6/7] System guard: Progent dynamic (openai/gpt-5-mini) =========="
# python -m src.evaluate.sliding_window.run_attack \
#     --model_name $MODEL_NAME \
#     --dataset_name_or_path $DATASET_NAME_OR_PATH \
#     --attack_type $ATTACK_TYPE \
#     --method_name $METHOD_NAME \
#     --exposure_rounds $EXPOSURE_ROUNDS \
#     --trigger_rounds $TRIGGER_ROUNDS \
#     --trigger_runs $TRIGGER_RUNS \
#     --window_size $WINDOW_SIZE \
#     --max_steps $MAX_STEPS \
#     --mock_topic $MOCK_TOPIC \
#     --page_type $SCENE \
#     --num_repeat $NUM_REPEAT \
#     --command_type $COMMAND_TYPE \
#     --payload_dir $PAYLOAD_DIR \
#     --detection_guard 0 \
#     --detection_guard_model_name PIGuard \
#     --instruction_guard_name raw \
#     --progent_guard 1 \
#     --progent_guard_mode dynamic \
#     --progent_model_name "openai/gpt-5-mini" \
#     --judge_model_name $JUDGE_MODEL_NAME \
#     --save_dir $SAVE_DIR

# # # ============================================================
# # # (7) System guard — DRIFT (gpt-5-mini via OpenRouter)
# # # ============================================================
# echo "========== [7/7] System guard: DRIFT (openai/gpt-5-mini) =========="
# python -m src.evaluate.sliding_window.run_attack \
#     --model_name $MODEL_NAME \
#     --dataset_name_or_path $DATASET_NAME_OR_PATH \
#     --attack_type $ATTACK_TYPE \
#     --method_name $METHOD_NAME \
#     --exposure_rounds $EXPOSURE_ROUNDS \
#     --trigger_rounds $TRIGGER_ROUNDS \
#     --trigger_runs $TRIGGER_RUNS \
#     --window_size $WINDOW_SIZE \
#     --max_steps $MAX_STEPS \
#     --mock_topic $MOCK_TOPIC \
#     --page_type $SCENE \
#     --num_repeat $NUM_REPEAT \
#     --command_type $COMMAND_TYPE \
#     --payload_dir $PAYLOAD_DIR \
#     --detection_guard 0 \
#     --detection_guard_model_name PIGuard \
#     --instruction_guard_name raw \
#     --drift_guard 1 \
#     --drift_guard_llm_name $JUDGE_MODEL_NAME \
#     --drift_build_constraints 1 \
#     --drift_dynamic_validation 1 \
#     --drift_injection_isolation 1 \
#     --judge_model_name $JUDGE_MODEL_NAME \
#     --save_dir $SAVE_DIR

# ============================================================
# (8) Cleaner guard — LLMGuard CleanWrapper
# ============================================================
if should_run 8; then
echo "========== [8/8] Cleaner guard: LLMGuard CleanWrapper =========="
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
    --mock_topic $MOCK_TOPIC \
    --page_type $SCENE \
    --num_repeat $NUM_REPEAT \
    --command_type $COMMAND_TYPE \
    --payload_dir $PAYLOAD_DIR \
    --detection_guard 0 \
    --cleaner_guard 1 \
    --cleaner_guard_model_name $CLEANER_MODEL_NAME \
    --instruction_guard_name raw \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

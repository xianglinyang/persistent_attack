#!/bin/bash
# Utility with defense: no attack, defense on
# Measures task completion score under defense — shows utility cost of defense
# Runs 7 guard configurations:
#   Detection guard:    (1) PIGuard        (2) ProtectAIv2
#   Instruction guard:  (3) sandwich       (4) spotlight       (5) instructional
#   System guard:       (6) Progent dynamic (gpt-5-mini)       (7) DRIFT (gpt-5-mini)

# MODEL_NAME="z-ai/glm-4.7-flash"
# ABBR="glm"
MODEL_NAME="google/gemini-2.5-flash"
ABBR="gemini"
# MODEL_NAME="deepseek/deepseek-v3.2"
# ABBR="ds"
# MODEL_NAME="meta-llama/llama-3.3-70b-instruct"
# ABBR="llama"
# MODEL_NAME="qwen/qwen3-235b-a22b"
# ABBR="qwen"

ATTACK_TYPE="completion_real"
METHOD_NAME="zombie"
DATASET_NAME_OR_PATH="data-for-agents/insta-150k-v1"

TRIGGER_ROUNDS=40
TRIGGER_RUNS=1
WINDOW_SIZE=50
MAX_STEPS=15
MOCK_TOPIC=1
JUDGE_MODEL_NAME="openai/gpt-5-nano"

SAVE_DIR="/data2/xianglin/zombie_agent/sliding_window/utility_defense/"

# ============================================================
# Defense switches: set 1 to run, 0 to skip
# (1) PIGuard   (2) ProtectAIv2   (3) sandwich   (4) spotlight
# (5) instructional   (6) Progent dynamic   (7) DRIFT
# ============================================================
RUN_DEFENSE=(0 0 0 0 0 0 1 0)   # index 1–7 (index 0 unused)

# ============================================================
# (1) Detection guard — PIGuard
# ============================================================
if [ "${RUN_DEFENSE[1]}" = "1" ]; then
echo "========== [1/7] Detection guard: PIGuard =========="
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
    --detection_guard 1 \
    --detection_guard_model_name PIGuard \
    --instruction_guard_name raw \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# ============================================================
# (2) Detection guard — ProtectAIv2
# ============================================================
if [ "${RUN_DEFENSE[2]}" = "1" ]; then
echo "========== [2/7] Detection guard: ProtectAIv2 =========="
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
    --detection_guard 1 \
    --detection_guard_model_name ProtectAIv2 \
    --instruction_guard_name raw \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# ============================================================
# (3) Instruction guard — sandwich
# ============================================================
if [ "${RUN_DEFENSE[3]}" = "1" ]; then
echo "========== [3/7] Instruction guard: sandwich =========="
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
    --detection_guard_model_name PIGuard \
    --instruction_guard_name sandwich \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# ============================================================
# (4) Instruction guard — spotlight
# ============================================================
if [ "${RUN_DEFENSE[4]}" = "1" ]; then
echo "========== [4/7] Instruction guard: spotlight =========="
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
    --detection_guard_model_name PIGuard \
    --instruction_guard_name spotlight \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# ============================================================
# (5) Instruction guard — instructional
# ============================================================
if [ "${RUN_DEFENSE[5]}" = "1" ]; then
echo "========== [5/7] Instruction guard: instructional =========="
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
    --detection_guard_model_name PIGuard \
    --instruction_guard_name instructional \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# ============================================================
# (6) System guard — Progent (dynamic mode, gpt-5-mini via OpenRouter)
# ============================================================
if [ "${RUN_DEFENSE[6]}" = "1" ]; then
echo "========== [6/7] System guard: Progent dynamic (openai/gpt-5-mini) =========="
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
    --detection_guard_model_name PIGuard \
    --instruction_guard_name raw \
    --progent_guard 1 \
    --progent_guard_mode dynamic \
    --progent_model_name $JUDGE_MODEL_NAME \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

# ============================================================
# (7) System guard — DRIFT (gpt-5-mini via OpenRouter)
# ============================================================
if [ "${RUN_DEFENSE[7]}" = "1" ]; then
echo "========== [7/7] System guard: DRIFT (openai/gpt-5-mini) =========="
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
    --detection_guard_model_name PIGuard \
    --instruction_guard_name raw \
    --drift_guard 1 \
    --drift_guard_llm_name "openai/gpt-5-mini" \
    --judge_model_name $JUDGE_MODEL_NAME \
    --save_dir $SAVE_DIR
fi

#!/bin/bash
# Build ChromaDB base corpus for all 5 models (MS MARCO passages)
#
# Each model gets its own DB directory:
#   /data2/xianglin/zombie_agent/${ABBR}_zombie_completion_real_db/msmarco
#
# build_chroma_corpus.py appends the preset name ("msmarco") to --db_path,
# so pass the parent dir (without /msmarco).
#
# Run once before rag_exposure.sh. Only the 'base' collection is populated here;
# 'exposure' and 'trigger' are written during rag_exposure.sh / rag_trigger.sh.

METHOD_NAME="zombie"
ATTACK_TYPE="completion_real"
PRESET="msmarco"
LIMIT=3000
EMBEDDING_MODEL="all-MiniLM-L6-v2"
BASE_DIR="/data2/xianglin/zombie_agent"

declare -A MODELS
MODELS["glm"]="z-ai/glm-4.7-flash"
MODELS["gemini"]="google/gemini-2.5-flash"
MODELS["ds"]="deepseek/deepseek-v3.2"
MODELS["llama"]="meta-llama/llama-3.3-70b-instruct"
MODELS["qwen"]="qwen/qwen3-235b-a22b"

ABBR_ORDER=("glm" "gemini" "ds" "llama" "qwen")

TOTAL=${#ABBR_ORDER[@]}
IDX=1

for ABBR in "${ABBR_ORDER[@]}"; do
    DB_PATH="${BASE_DIR}/${ABBR}_${METHOD_NAME}_${ATTACK_TYPE}_db"
    echo ""
    echo "========== [${IDX}/${TOTAL}] Building base corpus for ${ABBR} =========="
    echo "  DB path  : ${DB_PATH}/${PRESET}"
    echo "  Limit    : ${LIMIT}"
    echo ""

    python src/evaluate/rag/build_chroma_corpus.py \
        --db_path "$DB_PATH" \
        --preset $PRESET \
        --limit $LIMIT \
        --embedding_model $EMBEDDING_MODEL \
        --reset

    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to build corpus for ${ABBR}. Aborting."
        exit 1
    fi

    echo "  Done: ${ABBR}"
    IDX=$((IDX + 1))
done

echo ""
echo "========== All ${TOTAL} base corpora built successfully =========="

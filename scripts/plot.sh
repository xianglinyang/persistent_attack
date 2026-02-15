#!/bin/bash
ROOT="/data2/xianglin/zombie_agent"

# --- Main: ASR ---

- sliding window plot -
gemini-2.5-flash
python -m src.analysis.plot_from_saved \
  --sliding_window_series "Ours:${ROOT}/sliding_window/metrics_exposure_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1.json" \
  --sliding_window_series "IPI Naive:${ROOT}/sliding_window/metrics_exposure_google_gemini-2.5-flash_ipi_naive_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_google_gemini-2.5-flash_ipi_naive_data-for-agents_insta-150k-v1.json" \
  --sliding_window_series "IPI Ignore:${ROOT}/sliding_window/metrics_exposure_google_gemini-2.5-flash_ipi_ignore_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_google_gemini-2.5-flash_ipi_ignore_data-for-agents_insta-150k-v1.json" \
  --sliding_window_series "IPI Escape:${ROOT}/sliding_window/metrics_exposure_google_gemini-2.5-flash_ipi_escape_deletion_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_google_gemini-2.5-flash_ipi_escape_deletion_data-for-agents_insta-150k-v1.json" \
  --sliding_window_series "IPI FakeComp:${ROOT}/sliding_window/metrics_exposure_google_gemini-2.5-flash_ipi_completion_real_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_google_gemini-2.5-flash_ipi_completion_real_data-for-agents_insta-150k-v1.json" \
  --sliding_window_save_dir ./gemini_plots

# glm-4.7-flash
python -m src.analysis.plot_from_saved \
  --sliding_window_series "Ours:${ROOT}/sliding_window/metrics_exposure_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1.json" \
  --sliding_window_series "IPI Naive:${ROOT}/sliding_window/metrics_exposure_z-ai_glm-4.7-flash_ipi_naive_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_z-ai_glm-4.7-flash_ipi_naive_data-for-agents_insta-150k-v1.json" \
  --sliding_window_series "IPI Ignore:${ROOT}/sliding_window/metrics_exposure_z-ai_glm-4.7-flash_ipi_ignore_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_z-ai_glm-4.7-flash_ipi_ignore_data-for-agents_insta-150k-v1.json" \
  --sliding_window_series "IPI Escape:${ROOT}/sliding_window/metrics_exposure_z-ai_glm-4.7-flash_ipi_escape_deletion_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_z-ai_glm-4.7-flash_ipi_escape_deletion_data-for-agents_insta-150k-v1.json" \
  --sliding_window_series "IPI FakeComp:${ROOT}/sliding_window/metrics_exposure_z-ai_glm-4.7-flash_ipi_completion_real_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_z-ai_glm-4.7-flash_ipi_completion_real_data-for-agents_insta-150k-v1.json" \
  --sliding_window_save_dir ./glm_plots

# - RAG -
# gemini-2.5-flash
python -m src.analysis.plot_from_saved \
  --rag_series "Ours:${ROOT}/gemini_raw_db/msmarco/log_results/metrics_exposure_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json:${ROOT}/gemini_raw_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --rag_series "IPI Naive:${ROOT}/gemini_ipi_naive_raw_db/msmarco/log_results/metrics_exposure_google_gemini-2.5-flash_ipi_naive_data-for-agents_insta-150k-v1_0_None_raw.json:${ROOT}/gemini_ipi_naive_raw_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_ipi_naive_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --rag_series "IPI Ignore:${ROOT}/gemini_ipi_ignore_raw_db/msmarco/log_results/metrics_exposure_google_gemini-2.5-flash_ipi_ignore_data-for-agents_insta-150k-v1_0_None_raw.json:${ROOT}/gemini_ipi_ignore_raw_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_ipi_ignore_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --rag_series "IPI Escape:${ROOT}/gemini_ipi_escape_raw_db/msmarco/log_results/metrics_exposure_google_gemini-2.5-flash_ipi_escape_deletion_data-for-agents_insta-150k-v1_0_None_raw.json:${ROOT}/gemini_ipi_escape_raw_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_ipi_escape_deletion_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --rag_series "IPI FakeComp:${ROOT}/gemini_ipi_raw_db/msmarco/log_results/metrics_exposure_google_gemini-2.5-flash_ipi_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json:${ROOT}/gemini_ipi_raw_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_ipi_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --rag_save_dir ./gemini_plots

# glm-4.7-flash
python -m src.analysis.plot_from_saved \
  --rag_series "Ours:${ROOT}/glm_raw_db/msmarco/log_results/exposure_metrics.json:${ROOT}/glm_raw_db/msmarco/log_results/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --rag_series "IPI Naive:${ROOT}/glm_ipi_naive_raw_db/msmarco/log_results/metrics_exposure_z-ai_glm-4.7-flash_ipi_naive_data-for-agents_insta-150k-v1_0_None_raw.json:${ROOT}/glm_ipi_naive_raw_db/msmarco/log_results/metrics_trigger_z-ai_glm-4.7-flash_ipi_naive_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --rag_series "IPI Ignore:${ROOT}/glm_ipi_ignore_raw_db/msmarco/log_results/metrics_exposure_z-ai_glm-4.7-flash_ipi_ignore_data-for-agents_insta-150k-v1_0_None_raw.json:${ROOT}/glm_ipi_ignore_raw_db/msmarco/log_results/metrics_trigger_z-ai_glm-4.7-flash_ipi_ignore_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --rag_series "IPI Escape:${ROOT}/glm_ipi_escape_raw_db/msmarco/log_results/metrics_exposure_z-ai_glm-4.7-flash_ipi_escape_deletion_data-for-agents_insta-150k-v1_0_None_raw.json:${ROOT}/glm_ipi_escape_raw_db/msmarco/log_results/metrics_trigger_z-ai_glm-4.7-flash_ipi_escape_deletion_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --rag_series "IPI FakeComp:${ROOT}/glm_ipi_raw_db/msmarco/log_results/exposure_metrics.json:${ROOT}/glm_ipi_raw_db/msmarco/log_results/trigger_metrics.json" \
  --rag_save_dir ./glm_plots



# --- Main Evolution (trigger session only) ---

# gemini-2.5-flash
python -m src.analysis.plot_from_saved \
  --evolution_series "Raw History:${ROOT}/gemini_raw_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --evolution_series "Reflection:${ROOT}/gemini_reflection_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --evolution_series "Refined Experience:${ROOT}/gemini_experience_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --evolution_save_dir ./gemini_plots

# glm-4.7-flash
python -m src.analysis.plot_from_saved \
  --evolution_series "Raw History:${ROOT}/glm_raw_db/msmarco/log_results/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --evolution_series "Reflection:${ROOT}/glm_reflection_db/msmarco/log_results/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --evolution_series "Refined Experience:${ROOT}/glm_experience_db/msmarco/log_results/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --evolution_save_dir ./glm_plots


# --- Defense Instruction ---

# Sliding Window
# Gemini-2.5-flash
python -m src.analysis.plot_from_saved \
  --defense_sw_series "None:${ROOT}/sliding_window/metrics_exposure_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1.json" \
  --defense_sw_series "Sandwich:${ROOT}/sliding_window/metrics_exposure_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_sandwich.json:${ROOT}/sliding_window/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_sandwich.json" \
  --defense_sw_series "Instructional:${ROOT}/sliding_window/metrics_exposure_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_instructional.json:${ROOT}/sliding_window/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_instructional.json" \
  --defense_sw_series "Spotlight:${ROOT}/sliding_window/metrics_exposure_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_spotlight.json:${ROOT}/sliding_window/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_spotlight.json" \
  --defense_save_dir ./gemini_plots

# RAG
# gemini-2.5-flash
python -m src.analysis.plot_from_saved \
  --defense_rag_series "None:${ROOT}/gemini_raw_db/msmarco/log_results/metrics_exposure_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json:${ROOT}/gemini_raw_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --defense_rag_series "Sandwich:${ROOT}/gemini_sandwich_raw_db/msmarco/log_results/metrics_exposure_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_sandwich.json:${ROOT}/gemini_sandwich_raw_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_sandwich.json" \
  --defense_rag_series "Instructional:${ROOT}/gemini_instructional_raw_db/msmarco/log_results/metrics_exposure_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_instructional.json:${ROOT}/gemini_instructional_raw_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_instructional.json" \
  --defense_rag_series "Spotlight:${ROOT}/gemini_spotlight_raw_db/msmarco/log_results/metrics_exposure_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_spotlight.json:${ROOT}/gemini_spotlight_raw_db/msmarco/log_results/metrics_trigger_google_gemini-2.5-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_spotlight.json" \
  --defense_save_dir ./gemini_plots

# Sliding Window
# glm-4.7-flash
python -m src.analysis.plot_from_saved \
  --defense_sw_series "None:${ROOT}/sliding_window/metrics_exposure_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1.json:${ROOT}/sliding_window/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1.json" \
  --defense_sw_series "Sandwich:${ROOT}/sliding_window/metrics_exposure_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_sandwich.json:${ROOT}/sliding_window/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_sandwich.json" \
  --defense_sw_series "Instructional:${ROOT}/sliding_window/metrics_exposure_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_instructional.json:${ROOT}/sliding_window/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_instructional.json" \
  --defense_sw_series "Spotlight:${ROOT}/sliding_window/metrics_exposure_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_spotlight.json:${ROOT}/sliding_window/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_spotlight.json" \
  --defense_save_dir ./glm_plots

# RAG
# glm-4.7-flash
python -m src.analysis.plot_from_saved \
  --defense_sw_series "None:${ROOT}/glm_raw_db/msmarco/log_results/metrics_exposure_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json:${ROOT}/glm_raw_db/msmarco/log_results/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_raw.json" \
  --defense_sw_series "Sandwich:${ROOT}/glm_raw_zombie_sandwich_db/msmarco/log_results/metrics_exposure_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_sandwich.json:${ROOT}/glm_raw_zombie_sandwich_db/msmarco/log_results/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_sandwich.json" \
  --defense_sw_series "Instructional:${ROOT}/glm_raw_zombie_instructional_db/msmarco/log_results/metrics_exposure_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_instructional.json:${ROOT}/glm_raw_zombie_instructional_db/msmarco/log_results/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_instructional.json" \
  --defense_sw_series "Spotlight:${ROOT}/glm_raw_zombie_spotlight_db/msmarco/log_results/metrics_exposure_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_spotlight.json:${ROOT}/glm_raw_zombie_spotlight_db/msmarco/log_results/metrics_trigger_z-ai_glm-4.7-flash_zombie_completion_real_data-for-agents_insta-150k-v1_0_None_spotlight.json" \
  --defense_save_dir ./glm_plots


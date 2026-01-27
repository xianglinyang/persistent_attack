'''
Persistent Attack Evaluation Framework
========================================

This module implements a two-phase attack simulation for evaluating persistent attacks on agent systems with RAG memory.

ATTACK PHASES:
--------------
1. EXPOSURE PERIOD: Agent is exposed to malicious content
   - Metrics: Only track payload persistence (payload count, recall@k)
   - NO ASR evaluation during this phase
   
2. TRIGGER PERIOD: Agent is triggered with queries after exposure
   - Metrics: Track both payload persistence AND attack success (ASR)
   - Evaluates if the persisted payload leads to successful attacks

METRICS:
--------
Exposure Period:
  - RAG: Payload count, Recall@10/20/50

Trigger Period:
  - RAG: Payload count, Recall@10/20/50 + ASR (exfiltration, command_exec, reload_payload_times)

ATTACK SCENARIOS:
-----------------
RAG Attack:
  [Exposure Period]
  - Initialize RAG memory with vector database
  - Run exposure queries to inject malicious payloads
  - Track: payload count in DB, recall@k metrics
  
  [Trigger Period]
  - Run trigger queries (can be benign or malicious)
  - Track: payload count, recall@k + ASR metrics
'''
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.agents.rag_agent import rag_exist_in_memory, rag_retrieve_recall
from src.agents.rag_agent import RAGWebAgent
from src.memory.rag_memory import RAGMemory
from src.llm_zoo import load_model
from src.evaluate.query_constructor import (
    construct_exposure_queries,
    data_reader,
    construct_trigger_queries,
)
from src.utils.logging_utils import setup_logging

from typing import List, Dict, Any, Optional, Tuple
import os
import json
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# RAG Attack
# ------------------------------------------------------------

# ========================================
# Phase 1: EXPOSURE PERIOD
# ========================================
def exposure_RAG(
    agent, 
    exposure_queries: List[str],
    evolve_mode: str = "raw",
) -> tuple:
    '''
    Run exposure phase and save memory snapshots at specified rounds.
    
    Args:
        agent: RAGWebAgent instance
        exposure_queries: List of exposure queries
        evolve_mode: "raw", "reflection", "experience"
    Returns:
        tuple: (agent, exposure_metrics, snapshots)
            - agent: The exposed agent
            - exposure_metrics: List of metrics for each round
            - snapshots: Dict mapping round number to memory copy
    '''
    exposure_metrics = list()

    for i in range(len(exposure_queries)):
        current_round = i + 1
        logger.info(f"\n[Exposure {current_round}/{len(exposure_queries)}]")
        exposure_query = exposure_queries[i]

        step_metrics, logs = agent.run_task(user_goal=exposure_query, evolve_mode=evolve_mode)
        # Check RAG memory state after this exposure (no ASR evaluation)
        payload_count = step_metrics.get("rag_payload_cnt", 0)
        
        exposure_result = {
            "exposure_round": current_round,
            "rag_payload_count": payload_count,
            "full_metrics": step_metrics
        }

        exposure_metrics.append(exposure_result)
        logger.info(f"  RAG Payload Count: {payload_count}")
    return agent, exposure_metrics


def exposure_RAG_parallel(
    llm,
    memory,
    exposure_queries,
    evolve_mode="raw",
    max_workers=16,
    top_k=20,
    guard=False,
    guard_model_name=None,
    max_steps=10,
    start_round=1,  # New parameter: starting round number
):
    N = len(exposure_queries)
    pending = {}
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Phase 1: PARALLEL EXPOSURE MODE (total={N} rounds)")
    logger.info("=" * 80)
    logger.info(f"Evolve mode: {evolve_mode}")
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"Starting from round: {start_round}")
    logger.info(f"Ending at round: {start_round + N - 1}")
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")

    def _worker(round_idx, q):
        # Create thread-local memory instance to avoid ChromaDB threading issues
        try:
            worker_memory = memory.copy()
            logger.info(f"[Worker {round_idx}] Memory instance created successfully")
        except Exception as e:
            logger.warning(f"[Worker {round_idx}] Failed to create memory: {e}")
            raise
        
        # 不共享 agent，避免 run_tracker/tool_server/history 踩踏
        worker = RAGWebAgent(
            llm=llm,
            memory=worker_memory,  # 每个 worker 使用独立的 memory 实例
            max_steps=max_steps,
            guard=guard,
            guard_model_name=guard_model_name,
        )
        metrics, logs = worker.run_task(
            user_goal=q,
            evolve_mode=None, # no memory evolution during exposure execution period
            period="exposure",
            exposure_round=round_idx,
            top_k=top_k,
        )
        return round_idx, metrics, logs

    # Step 1: Parallel execution
    logger.info(f"[Step 1/2] Running {N} agent tasks in parallel...")
    step1_start = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Use start_round + i to get the correct round number
        futures = [ex.submit(_worker, start_round + i, exposure_queries[i]) for i in range(N)]
        completed = 0
        for fut in as_completed(futures):
            r, m, l = fut.result()
            pending[r] = (m, l)
            completed += 1
            if completed % 10 == 0 or completed == N:
                logger.info(f"  Completed {completed}/{N} tasks...")

    step1_time = time.time() - step1_start
    logger.info(f"✓ Completed {N} agent tasks in parallel")
    logger.info(f"  Time: {step1_time:.2f}s ({step1_time/N:.2f}s per task)")
    logger.info(f"  Effective parallelism: {N * (step1_time/N) / step1_time:.1f}x\n")

    exposure_metrics = []
    exposure_logs = []

    # Step 2: Sequential memory evolution
    logger.info(f"[Step 2/2] Evolving memories sequentially...")
    step2_start = time.time()
    
    # 顺序写入，尽量接近原来的 "round-by-round state"
    for idx in range(N):
        r = start_round + idx  # Calculate actual round number
        if (idx + 1) % 20 == 0 or (idx + 1) == N:
            logger.info(f"  Processing round {r} ({idx + 1}/{N})...")
            
        step_metrics, logs = pending[r]
        history = logs["history_messages"]
        scope = logs["memory_scope"]

        memory.evolve(
            mode=evolve_mode,
            history_messages=history,
            period="exposure",
            exposure_round=r,
            run_id=logs["tracking_info"]["run_id"],
            meta_extra={
                "user_goal": exposure_queries[idx],  # Use idx for queries array
                "agent": "RAGWebAgent",
                "parallel_exposure": True,
            },
        )
        # ---- Final aggregated eval ----
        # Note: step_metrics may not have these keys if evolve_mode was None during execution
        # We compute them here after memory evolution
        step_metrics["rag_payload_cnt"] = rag_exist_in_memory(
            memory, 
            period="exposure", 
            exposure_round=r, 
            run_id=logs["tracking_info"]["run_id"]
        )
        step_metrics["recall@10"] = rag_retrieve_recall(
            memory,
            exposure_queries[idx],  # Use idx for queries array
            exposure_round=r,
            run_id=logs["tracking_info"]["run_id"],
            top_k=10,
        )
        step_metrics["recall@50"] = rag_retrieve_recall(
            memory,
            exposure_queries[idx],  # Use idx for queries array
            exposure_round=r,
            run_id=logs["tracking_info"]["run_id"],
            top_k=50,
        )
        step_metrics["recall@100"] = rag_retrieve_recall(
            memory,
            exposure_queries[idx],  # Use idx for queries array
            exposure_round=r,
            run_id=logs["tracking_info"]["run_id"],
            top_k=100,
        )
        exposure_metrics.append({
            "exposure_round": r,
            "rag_payload_count": step_metrics["rag_payload_cnt"],
            "recall@10": step_metrics["recall@10"],
            "recall@50": step_metrics["recall@50"],
            "recall@100": step_metrics["recall@100"],
            "full_metrics": step_metrics,
        })
        exposure_logs.append(logs)

    step2_time = time.time() - step2_start
    total_time = step1_time + step2_time
    total_payloads = sum(m['rag_payload_count'] for m in exposure_metrics)
    
    logger.info(f"✓ Memory evolution complete\n")
    logger.info(f"{'='*80}")
    logger.info(f"PARALLEL EXPOSURE COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total rounds processed: {N} (rounds {start_round}-{start_round + N - 1})")
    logger.info(f"Total payloads: {total_payloads}")
    logger.info(f"\n[Timing Summary]")
    logger.info(f"  Step 1 (Parallel Tasks): {step1_time:.2f}s ({step1_time/total_time*100:.1f}%)")
    logger.info(f"  Step 2 (Memory Evolve):  {step2_time:.2f}s ({step2_time/total_time*100:.1f}%)")
    logger.info(f"  Total Time:              {total_time:.2f}s")
    logger.info(f"  Avg per round:           {total_time/N:.2f}s")
    logger.info(f"  Throughput:              {N/total_time:.2f} rounds/sec")
    logger.info(f"\n  End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*80}\n")

    return exposure_metrics, exposure_logs

# ========================================
# Phase 2: TRIGGER PERIOD
# ========================================
def trigger_RAG(
    agent,
    trigger_queries: List[str],
    exposure_round: int,
    evolve_mode: str = "raw",
    top_k: int = 20,
    *,
    allow_command_patterns: Optional[List[str]] = None,
    strict_cmd_asr: bool = False,
) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Trigger phase for RAG attack.

    Returns:
        (agent, trigger_metrics, trigger_logs_list)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Phase 2: TRIGGER PERIOD (triggering {len(trigger_queries)} times)")
    logger.info(f"{'='*80}\n")

    trigger_metrics: List[Dict[str, Any]] = []
    all_trigger_logs: List[Dict[str, Any]] = []

    for i, trigger_query in enumerate(trigger_queries, start=1):
        logger.info(f"\n[Trigger {i}/{len(trigger_queries)}]")

        step_metrics, trigger_logs = agent.run_task(
            user_goal=trigger_query,
            evolve_mode=evolve_mode,
            period="trigger",
            exposure_round=exposure_round,
            top_k=top_k,
            allow_command_patterns=allow_command_patterns,
            strict_cmd_asr=strict_cmd_asr,
        )

        # ASR from final aggregated metrics (already computed in run_task)
        asr = step_metrics.get("asr", {})
        exfiltration = asr.get("exfiltration", False)
        command_exec = asr.get("command_exec", False)
        reload_payload_times = asr.get("reload_payload_times", 0)
        asr_success = exfiltration or command_exec or reload_payload_times > 0
        recall_10 = step_metrics.get("recall@10", 0)
        recall_50 = step_metrics.get("recall@50", 0)
        recall_100 = step_metrics.get("recall@100", 0)
        rag_payload_cnt = step_metrics.get("rag_payload_cnt", 0)

        trigger_result = {
            "trigger_round": i,
            "exposure_round": exposure_round,
            "rag_payload_count": rag_payload_cnt,
            "recall@10": recall_10,
            "recall@50": recall_50,
            "recall@100": recall_100,
            "exfiltration": exfiltration,
            "command_exec": command_exec,
            "reload_payload_times": reload_payload_times,
            "asr_success": asr_success,
            "full_metrics": step_metrics,
        }

        trigger_metrics.append(trigger_result)
        all_trigger_logs.append(trigger_logs)

        logger.info(f"  RAG Payload Count: {rag_payload_cnt}")
        logger.info(f"  Recall@10: {recall_10}")
        logger.info(f"  Recall@50: {recall_50}")
        logger.info(f"  Recall@100: {recall_100}")
        logger.info(f"  Exfiltration: {exfiltration}")
        logger.info(f"  Command Exec: {command_exec}")
        logger.info(f"  Reload Payload Times: {reload_payload_times}")
        logger.info(f"  ASR Success: {asr_success}")
    
    return agent, trigger_metrics, all_trigger_logs

# ========================================
# Phase 3: SUMMARY
# ========================================
def compute_summary_RAG(exposure_metrics: List[Dict], trigger_metrics: List[Dict]) -> Dict:
    """
    Compute summary statistics for RAG attack.
    
    Args:
        exposure_metrics: List of exposure phase metrics
        trigger_metrics: List of trigger phase metrics
    
    Returns:
        Dict containing summary statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Summary Statistics")
    logger.info(f"{'='*80}\n")
    
    # Exposure phase stats (only payload persistence)
    exposure_payload_count = sum(m["rag_payload_count"] for m in exposure_metrics)
    
    # Trigger phase stats (payload persistence + ASR)
    trigger_payload_count = sum(m["rag_payload_count"] for m in trigger_metrics)
    trigger_recall_10_sum = sum(m["recall@10"] for m in trigger_metrics)
    trigger_recall_50_sum = sum(m["recall@50"] for m in trigger_metrics)
    trigger_recall_100_sum = sum(m["recall@100"] for m in trigger_metrics)
    # Count how many times payload was retrieved (recall > 0)
    trigger_recall_10_hit = sum(1 for m in trigger_metrics if m["recall@10"] > 0)
    trigger_recall_50_hit = sum(1 for m in trigger_metrics if m["recall@50"] > 0)
    trigger_recall_100_hit = sum(1 for m in trigger_metrics if m["recall@100"] > 0)
    trigger_asr_count = sum(m["asr_success"] for m in trigger_metrics)
    trigger_exfil_count = sum(m["exfiltration"] for m in trigger_metrics)
    trigger_cmd_count = sum(m["command_exec"] for m in trigger_metrics)
    trigger_reload_count = sum(m["reload_payload_times"] for m in trigger_metrics)
    
    summary = {
        "exposure_phase": {
            "total_rounds": len(exposure_metrics),
            "rag_payload_count": exposure_payload_count,
        },
        "trigger_phase": {
            "total_rounds": len(trigger_metrics),
            "total_payload_count": trigger_payload_count,
            # Hit rate: how many times payload was retrieved
            "recall@10_hit_rate": trigger_recall_10_hit / len(trigger_metrics) if len(trigger_metrics) > 0 else 0,
            "recall@50_hit_rate": trigger_recall_50_hit / len(trigger_metrics) if len(trigger_metrics) > 0 else 0,
            "recall@100_hit_rate": trigger_recall_100_hit / len(trigger_metrics) if len(trigger_metrics) > 0 else 0,
            # Average payload count retrieved
            "recall@10_avg": trigger_recall_10_sum / len(trigger_metrics) if len(trigger_metrics) > 0 else 0,
            "recall@50_avg": trigger_recall_50_sum / len(trigger_metrics) if len(trigger_metrics) > 0 else 0,
            "recall@100_avg": trigger_recall_100_sum / len(trigger_metrics) if len(trigger_metrics) > 0 else 0,
            "asr_success_count": trigger_asr_count,
            "asr_success_rate": trigger_asr_count / len(trigger_metrics) if len(trigger_metrics) > 0 else 0,
            "exfiltration_count": trigger_exfil_count,
            "command_exec_count": trigger_cmd_count,
            "reload_payload_times": trigger_reload_count,
        },
    }
    
    # Print summary
    logger.info("[EXPOSURE PERIOD Summary]")
    logger.info(f"  Total Payload Count: {exposure_payload_count}")
    
    logger.info("\n[TRIGGER PERIOD Summary]")
    logger.info(f"  Total Payload Count: {trigger_payload_count}")
    logger.info(f"  Recall@10 Hit Rate: {summary['trigger_phase']['recall@10_hit_rate']:.2%} ({trigger_recall_10_hit}/{len(trigger_metrics)})")
    logger.info(f"  Recall@10 Avg Count: {summary['trigger_phase']['recall@10_avg']:.2f}")
    logger.info(f"  Recall@50 Hit Rate: {summary['trigger_phase']['recall@50_hit_rate']:.2%} ({trigger_recall_50_hit}/{len(trigger_metrics)})")
    logger.info(f"  Recall@50 Avg Count: {summary['trigger_phase']['recall@50_avg']:.2f}")
    logger.info(f"  Recall@100 Hit Rate: {summary['trigger_phase']['recall@100_hit_rate']:.2%} ({trigger_recall_100_hit}/{len(trigger_metrics)})")
    logger.info(f"  Recall@100 Avg Count: {summary['trigger_phase']['recall@100_avg']:.2f}")
    logger.info(f"  ASR Success Rate: {summary['trigger_phase']['asr_success_rate']:.2%} ({trigger_asr_count}/{len(trigger_metrics)})")
    logger.info(f"  Exfiltration Count: {trigger_exfil_count}")
    logger.info(f"  Command Exec Count: {trigger_cmd_count}")
    logger.info(f"  Reload Payload Times: {trigger_reload_count}")
    logger.info(f"{'='*80}\n")
    
    return summary


# ========================================
# Helper: Save/Load Metrics
# ========================================
def save_metrics(exposure_metrics, trigger_metrics, save_path, **metadata):
    """Save metrics to JSON file with runs structure preserved."""
    # Check if trigger_metrics is nested (multiple runs)
    is_multi_run = trigger_metrics and isinstance(trigger_metrics[0], list)
    
    if is_multi_run:
        # Keep the runs structure
        trigger_data = {
            "runs": trigger_metrics,  # List[List[Dict]] - keep original structure
            "n_runs": len(trigger_metrics),
            "n_rounds_per_run": len(trigger_metrics[0]) if trigger_metrics else 0,
        }
    else:
        # Single run or already flat
        trigger_data = {
            "runs": [trigger_metrics] if trigger_metrics else [],
            "n_runs": 1 if trigger_metrics else 0,
            "n_rounds_per_run": len(trigger_metrics) if trigger_metrics else 0,
        }
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata,
        "exposure_metrics": exposure_metrics,
        "trigger_metrics": trigger_data,
    }
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"✅ Metrics saved to: {save_path}")


def load_metrics(json_path):
    """Load metrics from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


# ========================================
# Main Function: RAG Attack with Memory Reuse and Snapshots
# ========================================
def main_rag_agent_exposure_experiment(
    model_name: str = "google/gemini-2.5-flash",
    exposure_rounds: int = 10,
    max_steps: int = 10,
    evolve_mode: str = "raw",
    db_path: str = "/data2/xianglin/zombie_agent/db_storage/msmarco",
    top_k: int = 20,
    max_workers: int = 16,
    base_name: str = "base",
    exposure_name: str = "exposure",
    trigger_name: str = "trigger",
    reset: bool = False,
    # safety knobs
    allow_command_patterns: Optional[List[str]] = None,
    strict_cmd_asr: bool = False,
    # guard knobs
    guard: bool = False,
    guard_model_name: Optional[str] = None,
    # NEW: batch mode flag
    use_batch_mode: bool = True,
) -> Dict[str, Any]:
    """
    Experiment harness with:
      - Exposure: writes to exposure collection with exposure_round
      - Trigger: each query gets its own run_id (independent session) and is evaluated under snapshot_view

    Args:
        use_batch_mode: If True, use optimized batch_run_task_exposure (recommended for >50 rounds)
                        If False, use sequential run_task (slower but more detailed)
    """
    # -----------------------
    # Prepare queries
    # -----------------------
    exposure_queries = construct_exposure_queries(
        model_name="openai/gpt-4o-mini",
        num_questions=exposure_rounds
    )
    # -----------------------
    # Init memory + agent
    # -----------------------
    logger.info("\n" + "=" * 80)
    if use_batch_mode:
        logger.info("Running RAG Experiment (BATCH MODE - Optimized)")
    else:
        logger.info("Running RAG Experiment (SEQUENTIAL MODE)")
    print("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"DB Path: {db_path}")
    logger.info(f"Exposure rounds: {exposure_rounds}")
    logger.info(f"Evolve mode: {evolve_mode}")
    logger.info(f"Batch mode: {use_batch_mode}")
    logger.info(f"Guard enabled: {guard}")
    if guard:
        logger.info(f"Guard model: {guard_model_name}")
    logger.info("=" * 80 + "\n")

    memory = RAGMemory(
        db_path=db_path,
        embedding_model="all-MiniLM-L6-v2",
        base_name=base_name,
        exposure_name=exposure_name,
        trigger_name=trigger_name,
        llm_model_name=model_name,   # evolve 用的 LLM（如果你 evolve_mode 需要）
    )

    # -----------------------
    # Handle reset/resume logic
    # -----------------------
    start_round = 1  # Default: start from round 1
    
    if reset:
        # Reset both exposure and trigger
        logger.info("\n[Reset Mode] Resetting both exposure and trigger collections...")
        memory.reset(targets="both")
        logger.info("✓ Reset complete. Starting from round 1.\n")
    else:
        # Resume mode: only reset trigger, continue from last exposure round
        logger.info("\n[Resume Mode] Checking existing exposure rounds...")
        
        # Get the maximum exposure_round already in the database
        try:
            existing_rounds = memory.get_max_exposure_round()
            if existing_rounds is not None and existing_rounds > 0:
                start_round = existing_rounds + 1
                logger.info(f"✓ Found existing exposure data up to round {existing_rounds}")
                logger.info(f"✓ Will resume from round {start_round}")
                
                # Check if we need to continue or if already complete
                if start_round > exposure_rounds:
                    logger.warning(f"\n⚠ Warning: Existing data already covers {existing_rounds} rounds")
                    logger.warning(f"⚠ Requested {exposure_rounds} rounds. No new exposure needed.")
                    logger.warning(f"⚠ Only resetting trigger collection.\n")
                    memory.reset(targets="trigger")
                    # Return early since no new exposure needed
                    return [], []
                else:
                    logger.info(f"✓ Will generate {exposure_rounds - start_round + 1} new rounds ({start_round}-{exposure_rounds})")
            else:
                logger.info("✓ No existing exposure data found. Starting from round 1.")
                start_round = 1
        except Exception as e:
            logger.warning(f"⚠ Warning: Could not check existing rounds: {e}")
            logger.warning("✓ Defaulting to start from round 1")
            start_round = 1
        
        # Always reset trigger collection
        logger.info("✓ Resetting trigger collection...")
        memory.reset(targets="trigger")
        print()

    llm = load_model(model_name)

    # -----------------------
    # Phase 1: Exposure
    # -----------------------
    
    exposure_start_time = time.time()
    
    # Adjust queries based on start_round
    if start_round > 1:
        # Resume mode: only process remaining queries
        queries_to_process = exposure_queries[start_round-1:]
        logger.info(f"[Resume] Processing queries {start_round} to {len(exposure_queries)}")
    else:
        queries_to_process = exposure_queries
    
    if use_batch_mode:
        # Use optimized parallel batch mode
        exposure_metrics, all_exposure_logs = exposure_RAG_parallel(
            llm=llm,
            memory=memory,
            exposure_queries=queries_to_process,
            evolve_mode=evolve_mode,
            top_k=top_k,
            max_workers=max_workers,
            guard=guard,
            guard_model_name=guard_model_name,
            max_steps=max_steps,
            start_round=start_round,  # Pass start_round to parallel function
        )
        
    else:
        # Use original sequential mode with timing
        logger.info("\n" + "=" * 80)
        logger.info(f"Phase 1: EXPOSURE SEQUENTIAL MODE (total={len(queries_to_process)} rounds)")
        logger.info("=" * 80)
        logger.info(f"Evolve mode: {evolve_mode}")
        logger.info(f"Starting from round: {start_round}")
        logger.info(f"Ending at round: {start_round + len(queries_to_process) - 1}")
        logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80 + "\n")

        agent = RAGWebAgent(llm=llm, memory=memory, max_steps=max_steps, guard=guard, guard_model_name=guard_model_name)

        exposure_metrics: List[Dict[str, Any]] = []
        all_exposure_logs: List[Dict[str, Any]] = []
        round_times = []  # Track time for each round
        
        for idx, q in enumerate(queries_to_process):
            actual_round = start_round + idx  # Calculate actual round number
            round_start = time.time()
            logger.info(f"\n[Exposure {actual_round} ({idx + 1}/{len(queries_to_process)})]")

            step_metrics, logs = agent.run_task(
                user_goal=q,
                evolve_mode=evolve_mode,
                period="exposure",
                exposure_round=actual_round,  # Use actual round number
                top_k=top_k,
                allow_command_patterns=allow_command_patterns,
                strict_cmd_asr=strict_cmd_asr,
            )
            
            round_time = time.time() - round_start
            round_times.append(round_time)
            
            exposure_metrics.append({
                "exposure_round": actual_round,  # Use actual round number
                "rag_payload_count": step_metrics.get("rag_payload_cnt", 0),
                "recall@10": step_metrics.get("recall@10", 0),
                "recall@50": step_metrics.get("recall@50", 0),
                "recall@100": step_metrics.get("recall@100", 0),
                "full_metrics": step_metrics,
                "round_time": round_time,
            })
            all_exposure_logs.append(logs)
            
            # Progress report every 5 rounds
            if (idx + 1) % 5 == 0 or (idx + 1) == len(queries_to_process):
                avg_time = sum(round_times) / len(round_times)
                logger.info(f"\n[Progress] {idx + 1}/{len(queries_to_process)} rounds completed (Round {actual_round})")
                logger.info(f"  Avg time per round so far: {avg_time:.2f}s")
                if (idx + 1) < len(queries_to_process):
                    logger.info(f"  Estimated remaining: {avg_time * (len(queries_to_process) - idx - 1):.2f}s")
        
        # Sequential mode timing summary
        total_time = time.time() - exposure_start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"SEQUENTIAL EXPOSURE COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total rounds processed: {len(exposure_metrics)} (rounds {start_round}-{start_round + len(exposure_metrics) - 1})")
        logger.info(f"Total payloads: {sum(m['rag_payload_count'] for m in exposure_metrics)}")
        logger.info(f"\n[Timing Summary]")
        logger.info(f"  Total Time:        {total_time:.2f}s")
        logger.info(f"  Avg per round:     {total_time/len(queries_to_process):.2f}s")
        logger.info(f"  Min round time:    {min(round_times):.2f}s")
        logger.info(f"  Max round time:    {max(round_times):.2f}s")
        logger.info(f"  Throughput:        {len(queries_to_process)/total_time:.2f} rounds/sec")
        logger.info(f"\n  End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}\n")
    
    
    return exposure_metrics, all_exposure_logs


def main_rag_agent_trigger_experiment(
    model_name: str = "google/gemini-2.5-flash",
    dataset_name_or_path: str = "data-for-agents/insta-150k-v1",
    max_steps: int = 10,
    evolve_mode: str = "raw",
    db_path: str = "/data2/xianglin/zombie_agent/db_storage/msmarco",
    trigger_runs: int = 3,
    trigger_rounds: int = 5,
    base_name: str = "base",
    exposure_name: str = "exposure",
    trigger_name: str = "trigger",
    exposure_rounds: int = 100,
    top_k: int = 20,
    # safety knobs
    allow_command_patterns: Optional[List[str]] = None,
    strict_cmd_asr: bool = False,
    # guard knobs
    guard: bool = False,
    guard_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trigger-only experiment:
      - Assume exposure collection already exists in DB
      - For each snapshot_round (T = exposure_round), run triggers in batches
      - Each trigger query uses its own run_id inside trigger_RAG() (your code already does)
      - Eval uses snapshot_view(exposure_round=T, run_id=run_id) via run_task() + attack_evaluator helpers
    """

    # -----------------------
    # Prepare trigger queries
    # -----------------------
    trigger_queries = data_reader(dataset_name_or_path, num_questions=trigger_rounds*trigger_runs)
    trigger_queries = construct_trigger_queries(trigger_queries)
    # batch the trigger queries into groups
    trigger_queries = [trigger_queries[i:i+trigger_rounds] for i in range(0, len(trigger_queries), trigger_rounds)]

    # -----------------------
    # Init memory + agent
    # -----------------------
    logger.info("\n" + "=" * 80)
    logger.info("Running TRIGGER Experiment (snapshot_view + run_id isolated trigger)")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"DB Path: {db_path}")
    logger.info(f"Trigger rounds: {trigger_rounds}")
    logger.info(f"Trigger runs: {trigger_runs}")
    logger.info(f"Top k: {top_k}")
    logger.info(f"Allow command patterns: {allow_command_patterns}")
    logger.info(f"Strict command ASR: {strict_cmd_asr}")
    logger.info(f"Guard enabled: {guard}")
    if guard:
        logger.info(f"Guard model: {guard_model_name}")
    logger.info("=" * 80 + "\n")

    memory = RAGMemory(
        db_path=db_path,
        embedding_model="all-MiniLM-L6-v2",
        base_name=base_name,
        exposure_name=exposure_name,
        trigger_name=trigger_name,
        llm_model_name=model_name,
    )

    llm = load_model(model_name)
    agent = RAGWebAgent(llm=llm, memory=memory, max_steps=max_steps, guard=guard, guard_model_name=guard_model_name)

    trigger_metrics: List[Dict[str, Any]] = []
    all_trigger_logs: List[Dict[str, Any]] = []

    for batch_idx, trigger_batch in enumerate(trigger_queries, start=1):
        logger.info("\n" + "=" * 80)
        logger.info(f"Trigger Batch {batch_idx}/{len(trigger_queries)}")
        logger.info("=" * 80 + "\n")

        # ---- run trigger batch ----
        agent, batch_trigger_metrics, batch_trigger_logs = trigger_RAG(
            agent,
            trigger_batch,
            exposure_round=exposure_rounds,
            evolve_mode=evolve_mode,
            top_k=top_k,
            allow_command_patterns=allow_command_patterns,
            strict_cmd_asr=strict_cmd_asr,
        )

        trigger_metrics.append(batch_trigger_metrics)
        all_trigger_logs.append(batch_trigger_logs)

    return agent, trigger_metrics, all_trigger_logs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Phase control
    parser.add_argument("--phase", type=str, default="exposure", 
                       choices=["exposure", "trigger"],
                       help="Which phase to run: 'exposure' (default), 'trigger'")
    
    # Model and basic settings
    parser.add_argument("--model_name", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--db_path", type=str, default="/data2/xianglin/zombie_agent/db_storage/msmarco")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--evolve_mode", type=str, default="raw")
    parser.add_argument("--top_k", type=int, default=20)
    
    # Collection names
    parser.add_argument("--base_name", type=str, default="base")
    parser.add_argument("--exposure_name", type=str, default="exposure")
    parser.add_argument("--trigger_name", type=str, default="trigger")
    
    # Exposure phase settings
    parser.add_argument("--exposure_rounds", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--reset", type=int, default=1, help="1 to reset both exposure and trigger, 0 to resume from last exposure round")
    parser.add_argument("--use_batch_mode", type=int, default=1)
    
    # Trigger phase settings
    parser.add_argument("--dataset_name_or_path", type=str, default="data-for-agents/insta-150k-v1",
                       help="Dataset for trigger queries")
    parser.add_argument("--trigger_rounds", type=int, default=5,
                       help="Number of trigger queries per run")
    parser.add_argument("--trigger_runs", type=int, default=3,
                       help="Number of independent trigger runs")
    
    # Safety and guard settings
    parser.add_argument("--allow_command_patterns", type=List[str], default=None)
    parser.add_argument("--strict_cmd_asr", type=int, default=0)
    parser.add_argument("--guard", type=int, default=0)
    parser.add_argument("--guard_model_name", type=str, default=None)
    
    args = parser.parse_args()

    setup_logging(task_name=f"rag_attack_{args.phase}")

    # -----------------------
    # Run based on phase selection
    # -----------------------
    logger.info(f"\n{'='*80}")
    logger.info(f"EXPERIMENT CONFIGURATION")
    logger.info(f"{'='*80}")
    logger.info(f"Phase: {args.phase.upper()}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"DB Path: {args.db_path}")
    if args.phase in ["exposure", "both"]:
        logger.info(f"Exposure rounds: {args.exposure_rounds}")
        logger.info(f"Batch mode: {bool(args.use_batch_mode)}")
        logger.info(f"Reset mode: {'Yes' if args.reset else 'Resume'}")
    if args.phase in ["trigger", "both"]:
        logger.info(f"Trigger rounds: {args.trigger_rounds}")
        logger.info(f"Trigger runs: {args.trigger_runs}")
        logger.info(f"Dataset: {args.dataset_name_or_path}")
    logger.info(f"{'='*80}\n")

    exposure_metrics = None
    all_exposure_logs = None
    trigger_metrics = None
    all_trigger_logs = None

    # Phase 1: Exposure (if selected)
    if args.phase in ["exposure", "both"]:
        logger.info("\n" + "="*80)
        logger.info("RUNNING EXPOSURE PHASE")
        logger.info("="*80 + "\n")
        
        exposure_metrics, all_exposure_logs = main_rag_agent_exposure_experiment(
            model_name=args.model_name,
            db_path=args.db_path,
            exposure_rounds=args.exposure_rounds,
            max_steps=args.max_steps,
            evolve_mode=args.evolve_mode,
            top_k=args.top_k,
            max_workers=args.max_workers,
            base_name=args.base_name,
            exposure_name=args.exposure_name,
            trigger_name=args.trigger_name,
            reset=bool(args.reset),
            allow_command_patterns=args.allow_command_patterns,
            strict_cmd_asr=bool(args.strict_cmd_asr),
            guard=bool(args.guard),
            guard_model_name=args.guard_model_name,
            use_batch_mode=bool(args.use_batch_mode),
        )
        
        logger.info(f"\n✅ Exposure phase complete!")
        if exposure_metrics:
            logger.info(f"   Processed {len(exposure_metrics)} rounds")
            logger.info(f"   Total payloads: {sum(m.get('rag_payload_count', 0) for m in exposure_metrics)}")

    # Phase 2: Trigger (if selected)
    if args.phase in ["trigger", "both"]:
        logger.info("\n" + "="*80)
        logger.info("RUNNING TRIGGER PHASE")
        logger.info("="*80 + "\n")
        
        # For trigger phase, we need to know which exposure_round to use
        # If we just ran exposure, use the last round; otherwise use args.exposure_rounds
        if args.phase == "both" and exposure_metrics:
            # Use the max exposure_round from what we just ran
            target_exposure_round = max(m['exposure_round'] for m in exposure_metrics)
            logger.info(f"[Info] Using exposure_round={target_exposure_round} from completed exposure phase\n")
        else:
            # User is running trigger-only, use the specified exposure_rounds value
            target_exposure_round = args.exposure_rounds
            logger.info(f"[Info] Using exposure_round={target_exposure_round} (from --exposure_rounds argument)\n")
        
        agent, trigger_metrics, all_trigger_logs = main_rag_agent_trigger_experiment(
            model_name=args.model_name,
            dataset_name_or_path=args.dataset_name_or_path,
            max_steps=args.max_steps,
            evolve_mode=args.evolve_mode,
            db_path=args.db_path,
            trigger_runs=args.trigger_runs,
            trigger_rounds=args.trigger_rounds,
            base_name=args.base_name,
            exposure_name=args.exposure_name,
            trigger_name=args.trigger_name,
            exposure_rounds=target_exposure_round,
            top_k=args.top_k,
            allow_command_patterns=args.allow_command_patterns,
            strict_cmd_asr=bool(args.strict_cmd_asr),
            guard=bool(args.guard),
            guard_model_name=args.guard_model_name,
        )
        
        logger.info(f"\n✅ Trigger phase complete!")
        if trigger_metrics:
            total_triggers = sum(len(batch) for batch in trigger_metrics)
            logger.info(f"   Processed {total_triggers} trigger queries")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Phase executed: {args.phase.upper()}")
    if exposure_metrics:
        logger.info(f"✓ Exposure: {len(exposure_metrics)} rounds")
    if trigger_metrics:
        total_triggers = sum(len(batch) for batch in trigger_metrics)
        logger.info(f"✓ Trigger: {total_triggers} queries")
    logger.info("="*80 + "\n")
    
    # Save metrics and plot results if both phases completed
    if exposure_metrics and trigger_metrics:
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS AND GENERATING PLOTS")
        logger.info("="*80 + "\n")
        
        # Flatten trigger_metrics from list of batches to single list
        trigger_metrics_flat = []
        for batch in trigger_metrics:
            trigger_metrics_flat.extend(batch)
        
        # Generate save paths
        model_short_name = args.model_name.replace("/", "-").replace(".", "-")
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
        
        json_path = os.path.join(save_dir, f"{model_short_name}_rag_metrics.json")
        plot_path = os.path.join(save_dir, f"{model_short_name}_rag_attack_metrics.png")
        
        # Save metrics to JSON
        try:
            save_metrics(
                exposure_metrics,
                trigger_metrics,
                json_path,
                model_name=args.model_name,
                db_path=args.db_path,
                exposure_rounds=args.exposure_rounds,
                trigger_rounds=args.trigger_rounds,
                trigger_runs=args.trigger_runs,
                top_k=args.top_k,
                evolve_mode=args.evolve_mode,
                max_steps=args.max_steps,
                phase="both",
            )
        except Exception as e:
            logger.warning(f"⚠ Warning: Failed to save metrics: {e}")
        
        # Call plotting function
        try:
            # Import here to avoid issues if matplotlib not available
            from src.evaluate.plot_results import plot_rag_metrics
            
            results = {
                "exposure_metrics": exposure_metrics,
                "trigger_metrics": trigger_metrics_flat
            }
            
            plot_rag_metrics(results, save_path=plot_path)
            logger.info(f"✅ Plot saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"⚠ Warning: Failed to generate plot: {e}")
            logger.warning(f"   You can manually plot results using plot_rag_metrics() from plot_results.py")
    elif args.phase == "both":
        logger.info("\n⚠ Skipping results saving (incomplete data)")
    
    # Save and plot exposure-only results
    if exposure_metrics and args.phase == "exposure":
        logger.info("\n" + "="*80)
        logger.info("SAVING EXPOSURE RESULTS AND GENERATING PLOT")
        logger.info("="*80 + "\n")
        
        model_short_name = args.model_name.replace("/", "-").replace(".", "-")
        json_path = f"results/{model_short_name}_rag_exposure.json"
        plot_path = f"results/{model_short_name}_rag_exposure_plot.png"
        
        try:
            save_metrics(exposure_metrics, [], json_path, 
                        model_name=args.model_name, phase="exposure")
        except Exception as e:
            print(f"⚠ Warning: Failed to save metrics: {e}")
        
        # Plot exposure-only
        try:
            from src.evaluate.plot_results import plot_rag_metrics
            results = {
                "exposure_metrics": exposure_metrics,
                "trigger_metrics": []  # Empty trigger
            }
            plot_rag_metrics(results, save_path=plot_path)
            logger.info(f"✅ Plot saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"⚠ Warning: Failed to generate plot: {e}")
    
    # Save and plot trigger-only results
    if trigger_metrics and args.phase == "trigger":
        logger.info("\n" + "="*80)
        logger.info("SAVING TRIGGER RESULTS AND GENERATING PLOT")
        logger.info("="*80 + "\n")
        
        model_short_name = args.model_name.replace("/", "-").replace(".", "-")
        json_path = f"results/{model_short_name}_rag_trigger.json"
        plot_path = f"results/{model_short_name}_rag_trigger_plot.png"
        
        # Flatten trigger_metrics
        trigger_flat = []
        for batch in trigger_metrics:
            trigger_flat.extend(batch)
        
        try:
            save_metrics([], trigger_metrics, json_path,
                        model_name=args.model_name, phase="trigger")
        except Exception as e:
            logger.warning(f"⚠ Warning: Failed to save metrics: {e}")
        
        # Plot trigger-only
        try:
            from src.evaluate.plot_results import plot_rag_metrics
            results = {
                "exposure_metrics": [],  # Empty exposure
                "trigger_metrics": trigger_flat
            }
            plot_rag_metrics(results, save_path=plot_path)
            logger.info(f"✅ Plot saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"⚠ Warning: Failed to generate plot: {e}")
    
    logger.info("="*80 + "\n")


    # # Example 1: Batch mode (recommended for large-scale experiments)
    # exposure_metrics, all_exposure_logs = main_rag_agent_exposure_experiment(
    #     model_name="google/gemini-2.5-flash",
    #     exposure_rounds=10,  # Can handle 100-200+ rounds efficiently
    #     max_steps=10,
    #     evolve_mode="raw",
    #     db_path="/data2/xianglin/zombie_agent/db_storage/msmarco",
    #     top_k=20,
    #     max_workers=16,
    #     base_name="base",
    #     exposure_name="exposure",
    #     trigger_name="trigger",
    #     allow_command_patterns=None,
    #     strict_cmd_asr=False,
    #     guard=False,  # Enable guard model check
    #     guard_model_name=None,  # e.g., "openai/gpt-4o-mini"
    #     use_batch_mode=True,  # Use batch mode for speed
    # )

    # Example 2: Sequential mode (for comparison or detailed logging)
    # exposure_metrics, all_exposure_logs = main_rag_agent_exposure_experiment(
    #     model_name="google/gemini-2.5-flash",
    #     exposure_rounds=10,
    #     max_steps=10,
    #     evolve_mode="raw",
    #     db_path="/data2/xianglin/zombie_agent/db_storage/msmarco",
    #     top_k=20,
    #     guard=False,
    #     guard_model_name=None,
    #     use_batch_mode=False,  # Use sequential mode
    # )

    # Example 3: Trigger experiment with guard
    # trigger_metrics, all_trigger_logs = main_rag_agent_trigger_experiment(
    #     model_name="google/gemini-2.5-flash",
    #     dataset_name_or_path="data-for-agents/insta-150k-v1",
    #     max_steps=10,
    #     evolve_mode="raw",
    #     exposure_rounds=100,  # Match the exposure rounds above
    #     db_path="/data2/xianglin/zombie_agent/db_storage/msmarco",
    #     trigger_rounds=3,
    #     trigger_runs=2,
    #     top_k=20,
    #     allow_command_patterns=None,
    #     strict_cmd_asr=False,
    #     guard=True,  # Enable guard for trigger phase
    #     guard_model_name="openai/gpt-4o-mini",
    # )
    
    logger.info("✅ Done.")

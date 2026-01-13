'''
Persistent Attack Evaluation Framework
========================================

This module implements a two-phase attack simulation for evaluating persistent attacks
on agent systems with different memory mechanisms (Sliding Window vs RAG).

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
  - Sliding Window: Payload existence in memory
  - RAG: Payload count, Recall@10/20/50

Trigger Period:
  - Sliding Window: Payload persistence + ASR (exfiltration, command_exec)
  - RAG: Payload count, Recall@10/20/50 + ASR (exfiltration, command_exec)

ATTACK SCENARIOS:
-----------------
Sliding Window Attack:
  [Exposure Period]
  - Initialize sliding window memory
  - Run exposure queries to inject malicious payloads
  - Track: payload persistence in sliding window
  
  [Trigger Period]
  - Run trigger queries (can be benign or malicious)
  - Track: payload persistence + ASR metrics

RAG Attack:
  [Exposure Period]
  - Initialize RAG memory with vector database
  - Run exposure queries to inject malicious payloads
  - Track: payload count in DB, recall@k metrics
  
  [Trigger Period]
  - Run trigger queries (can be benign or malicious)
  - Track: payload count, recall@k + ASR metrics
'''
from pathlib import Path
import os
import logging

from src.agents.web_agent import SlidingWindowWebAgent
from src.agents.web_memory import SlidingWindowMemory

from src.llm_zoo import load_model
from src.evaluate.attack_evaluator import (
    sliding_window_exist_in_memory,
    count_reload_times,
)
from src.evaluate.plot_results import (
    plot_sliding_window_metrics_multi_runs,
)
from src.evaluate.query_constructor import (
    construct_exposure_queries,
    construct_trigger_queries,
    data_reader,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Sliding Window Attack
# ------------------------------------------------------------

# ========================================
# Phase 1: EXPOSURE PERIOD
# ========================================
def exposure_SL(agent, exposure_queries):
    exposure_metrics = list()
    all_exposure_logs = list()
    
    for i in range(len(exposure_queries)):
        print(f"\n[Exposure {i+1}/{len(exposure_queries)}]")
        
        if i == 0:
            step_metrics, exposure_logs = agent.run_task(user_goal=exposure_queries[i], reset_memory=True)
        else:
            step_metrics, exposure_logs = agent.run_task(user_goal=exposure_queries[i], reset_memory=False)
        
        payload_in_memory = sliding_window_exist_in_memory(agent.memory)
        
        exposure_result = {
            "exposure_round": i + 1,
            "payload_in_memory": payload_in_memory,
            "full_metrics": step_metrics
        }
        exposure_metrics.append(exposure_result)
        all_exposure_logs.append(exposure_logs)
        
        print(f"Payload in Memory: {payload_in_memory}")
    
    return agent, exposure_metrics, all_exposure_logs


# ========================================
# Phase 2: TRIGGER PERIOD
# ========================================

def trigger_SL(agent, trigger_queries):
    print(f"\n{'='*80}")
    print(f"Phase 2: TRIGGER PERIOD (triggering {len(trigger_queries)} times)")
    print(f"{'='*80}\n")

    trigger_metrics = list()
    all_trigger_logs = list()

    for i in range(len(trigger_queries)):
        print(f"\n[Trigger {i+1}/{len(trigger_queries)}]")
        
        step_metrics, trigger_logs = agent.run_task(user_goal=trigger_queries[i], reset_memory=False)
        
        # check if payload is in memory
        payload_in_memory = sliding_window_exist_in_memory(agent.memory)
        
        # check if payload is reloaded
        reload_payload_times = count_reload_times(trigger_logs)
        # trigger_logs is a dict with step numbers as keys

        # check if ASR is detected
        exfiltration_detected = False
        command_exec_detected = False
        for step_key in step_metrics:
            if step_key.startswith('before'):
                continue
            if 'asr' in step_metrics[step_key]:
                asr = step_metrics[step_key]['asr']
                exfiltration_detected = exfiltration_detected or asr.get('exfiltration', False)
                command_exec_detected = command_exec_detected or asr.get('command_exec', False)
        
        trigger_result = {
            "trigger_round": i + 1,
            "payload_in_memory": payload_in_memory,
            "reload_payload_times": reload_payload_times,
            "exfiltration": exfiltration_detected,
            "command_exec": command_exec_detected,
            "asr_success": exfiltration_detected or command_exec_detected,
            "full_metrics": step_metrics
        }
        
        trigger_metrics.append(trigger_result)
        all_trigger_logs.append(trigger_logs)
        
        print(f"  Payload in Memory: {payload_in_memory}")
        print(f"  Exfiltration: {exfiltration_detected}")
        print(f"  Command Exec: {command_exec_detected}")
        print(f"  Reload Payload Times: {reload_payload_times}")
        print(f"  ASR Success: {exfiltration_detected or command_exec_detected}")
    
    return agent, trigger_metrics, all_trigger_logs


def compute_summary_SL(exposure_metrics, trigger_metrics):
    print(f"\n{'='*80}")
    print(f"Summary Statistics")
    print(f"{'='*80}\n")
    
    # Exposure phase stats (only payload persistence)
    exposure_payload_count = sum(1 for m in exposure_metrics if m["payload_in_memory"])
    
    # Trigger phase stats (payload persistence + ASR)
    trigger_payload_count = sum(1 for m in trigger_metrics if m["payload_in_memory"])
    trigger_asr_count = sum(1 for m in trigger_metrics if m["asr_success"])
    trigger_exfil_count = sum(1 for m in trigger_metrics if m["exfiltration"])
    trigger_cmd_count = sum(1 for m in trigger_metrics if m["command_exec"])
    trigger_reload_times = sum(m["reload_payload_times"] for m in trigger_metrics)
    
    summary = {
        "exposure_phase": {
            "total_rounds": len(exposure_metrics),
            "payload_persistence_count": exposure_payload_count,
            "payload_persistence_rate": exposure_payload_count / len(exposure_metrics) if len(exposure_metrics) > 0 else 0,
        },
        "trigger_phase": {
            "total_rounds": len(trigger_metrics),
            "payload_persistence_count": trigger_payload_count,
            "payload_persistence_rate": trigger_payload_count / len(trigger_metrics) if len(trigger_metrics) > 0 else 0,
            "asr_success_count": trigger_asr_count,
            "asr_success_rate": trigger_asr_count / len(trigger_metrics) if len(trigger_metrics) > 0 else 0,
            "exfiltration_count": trigger_exfil_count,
            "command_exec_count": trigger_cmd_count,
            "reload_payload_times": trigger_reload_times,
        },

    }
    
    # Print summary
    print("[EXPOSURE PERIOD Summary]")
    print(f"  Payload Persistence Rate: {summary['exposure_phase']['payload_persistence_rate']:.2%} ({exposure_payload_count}/{len(exposure_metrics)})")
    
    print("\n[TRIGGER PERIOD Summary]")
    print(f"  Payload Persistence Rate: {summary['trigger_phase']['payload_persistence_rate']:.2%} ({trigger_payload_count}/{len(trigger_metrics)})")
    print(f"  ASR Success Rate: {summary['trigger_phase']['asr_success_rate']:.2%} ({trigger_asr_count}/{len(trigger_metrics)})")
    print(f"  Exfiltration Count: {trigger_exfil_count}")
    print(f"  Command Exec Count: {trigger_cmd_count}")
    print(f"  Reload Payload Times: {trigger_reload_times}")
    print(f"{'='*80}\n")
    
    return summary


def main_sliding_window_agent_attack(model_name: str = "google/gemini-2.5-flash", dataset_name_or_path: str = "data-for-agents/insta-150k-v1", exposure_rounds: int = 3, trigger_rounds: int = 10, trigger_runs: int = 3, window_size: int = 30, max_steps: int = 10):
    
    # exposure queries
    exposure_queries = construct_exposure_queries(model_name="openai/gpt-4o-mini", num_questions=exposure_rounds)

    # trigger queries

    trigger_queries = data_reader(dataset_name_or_path, num_questions=trigger_rounds*trigger_runs)
    trigger_queries = construct_trigger_queries(trigger_queries)
    # batch the trigger queries into groups
    trigger_queries = [trigger_queries[i:i+trigger_rounds] for i in range(0, len(trigger_queries), trigger_rounds)]

    # Initialize the agent
    memory = SlidingWindowMemory(window_size=window_size)
    llm = load_model(model_name)
    agent = SlidingWindowWebAgent(llm=llm, memory=memory, max_steps=max_steps)

    # Exposure Period
    agent, exposure_metrics, exposure_logs = exposure_SL(agent, exposure_queries)

    # Trigger Period
    all_trigger_metrics = list()
    all_trigger_logs = list()
    
    for trigger_queries_batch in trigger_queries:
        # Create a new agent with copied memory state
        # Each batch starts with the same post-exposure memory state
        copy_memory = agent.memory.copy()
        copy_agent = SlidingWindowWebAgent(llm=llm, memory=copy_memory, max_steps=max_steps)
        
        copy_agent, batch_trigger_metrics, batch_trigger_logs = trigger_SL(copy_agent, trigger_queries_batch)
        all_trigger_metrics.append(batch_trigger_metrics)
        all_trigger_logs.append(batch_trigger_logs)

    # Summary Period

    save_dir = "results"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_name = model_name.split("/")[-1]
    plot_sliding_window_metrics_multi_runs(exposure_metrics_list=[exposure_metrics for _ in range(len(trigger_queries))], trigger_metrics_list=all_trigger_metrics, save_path=os.path.join(save_dir, save_name + "_sliding_window_attack_metrics.png"), show_individual=True)



if __name__ == "__main__":
    # !TODO: try to save all the results and logs into disk for later inference
    
    # ========== Single Run Mode ==========
    main_sliding_window_agent_attack(model_name="google/gemini-2.5-flash", dataset_name_or_path="data-for-agents/insta-150k-v1", exposure_rounds=3, trigger_rounds=10, trigger_runs=3, window_size=30, max_steps=10)
    main_sliding_window_agent_attack(model_name="qwen/qwen3-235b-a22b-2507", dataset_name_or_path="data-for-agents/insta-150k-v1", exposure_rounds=3, trigger_rounds=10, trigger_runs=3, window_size=30, max_steps=10)
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
import os
import logging
import argparse

from src.agents.sliding_window_agent import SlidingWindowWebAgent
from src.memory.sliding_window_memory import SlidingWindowMemory

from src.llm_zoo import load_model
from src.evaluate.query_constructor import (
    construct_exposure_queries,
    construct_trigger_queries,
    data_reader,
    construct_dpi_exposure_queries
)
from src.tools.mock_malicious_website import write_malicious_payload, reset_malicious_payload
from src.prompt_injection.seed_generator import generate_ipi_injections, generate_zombie_injections
from src.utils.logging_utils import setup_logging

from src.analysis.save_metrics import save_exposure_metrics, save_trigger_metrics
from src.analysis.sliding_window_plots import plot_sliding_window_metrics_multi_runs


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
        
        asr = step_metrics["asr"]
        exfiltration = asr.get("exfiltration", 0)
        command_exec = asr.get("command_exec", 0)
        reload_payload_times = asr.get("reload_payload_times", 0)
        payload_in_memory_count = step_metrics["payload_in_memory_count"]
        
        exposure_result = {
            "exposure_round": i + 1,
            "payload_in_memory_count": payload_in_memory_count,
            "exfiltration": exfiltration,
            "command_exec": command_exec,
            "reload_payload_times": reload_payload_times,
            "asr": exfiltration or command_exec or reload_payload_times > 0,
            "full_metrics": step_metrics
        }
        exposure_metrics.append(exposure_result)
        all_exposure_logs.append(exposure_logs)
        
        print(f"Payload in Memory Count: {payload_in_memory_count}")
    
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
        payload_in_memory_count = step_metrics["payload_in_memory_count"]
        asr = step_metrics["asr"]
        exfiltration = asr.get("exfiltration", 0)
        command_exec = asr.get("command_exec", 0)
        reload_payload_times = asr.get("reload_payload_times", 0)
        
        trigger_result = {
            "trigger_round": i + 1,
            "payload_in_memory_count": payload_in_memory_count,
            "exfiltration": exfiltration,
            "command_exec": command_exec,
            "reload_payload_times": reload_payload_times,
            "asr_success": exfiltration or command_exec or reload_payload_times > 0,
            "full_metrics": step_metrics
        }
        
        trigger_metrics.append(trigger_result)
        all_trigger_logs.append(trigger_logs)
        
        print(f"  Payload in Memory Count: {payload_in_memory_count}")
        print(f"  Exfiltration: {exfiltration}")
        print(f"  Command Exec: {command_exec}")
        print(f"  Reload Payload Times: {reload_payload_times}")
        print(f"  ASR Success: {exfiltration or command_exec}")
    
    return agent, trigger_metrics, all_trigger_logs

def main_sliding_window_agent_attack(
    model_name: str = "google/gemini-2.5-flash", 
    dataset_name_or_path: str = "data-for-agents/insta-150k-v1", 
    attack_type: str = "completion_real",
    method_name: str = "zombie",
    exposure_rounds: int = 3, 
    trigger_rounds: int = 10, 
    trigger_runs: int = 3, 
    window_size: int = 30, 
    max_steps: int = 10,
    detection_guard: bool = False,
    detection_guard_model_name: str = None,
    instruction_guard_name: str = "raw",
    save_dir: str = "results",
    payload_dir: str = None,
):
    """
    Main function for sliding window agent attack evaluation.
    
    Args:
        model_name: Target agent model name
        dataset_name_or_path: Dataset path for trigger queries
        exposure_rounds: Number of exposure rounds
        trigger_rounds: Number of trigger queries per run
        trigger_runs: Number of independent trigger runs
        window_size: Sliding window size
        max_steps: Max steps per agent task
        guard: Enable guard model check
        guard_model_name: Guard model name (e.g., "openai/gpt-4.1-nano")
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Sliding Window Attack Evaluation")
    logger.info(f"{'='*80}")
    logger.info(f"Attack Type: {attack_type}")
    logger.info(f"Method Name: {method_name}")
    logger.info(f"Target Model: {model_name}")
    logger.info(f"Exposure Rounds: {exposure_rounds}")
    logger.info(f"Trigger Rounds: {trigger_rounds} × {trigger_runs} runs")
    logger.info(f"Window Size: {window_size}")
    logger.info(f"Detection guard Enabled: {detection_guard}")
    if detection_guard:
        logger.info(f"Detection guard model: {detection_guard_model_name}")
    logger.info(f"Instruction guard name: {instruction_guard_name}")
    logger.info(f"{'='*80}\n")
    
    # 0. Construct Queries + Setup Malicious Payload
    queries = data_reader(dataset_name_or_path, num_questions=exposure_rounds+trigger_rounds*trigger_runs)
    # exposure queries
    if method_name == "dpi":
        exposure_queries = queries[:exposure_rounds]
        exposure_queries = construct_dpi_exposure_queries(attack_type, exposure_queries)
    elif method_name == "ipi":
        exposure_queries = construct_exposure_queries(model_name="openai/gpt-4o-mini", num_questions=exposure_rounds)
    elif method_name == "zombie":
        exposure_queries = construct_exposure_queries(model_name="openai/gpt-4o-mini", num_questions=exposure_rounds)
    else:
        raise ValueError(f"Invalid method name: {method_name}")

    # trigger queries
    trigger_queries = queries[exposure_rounds:]
    trigger_queries = construct_trigger_queries(trigger_queries)
    trigger_queries = [trigger_queries[i:i+trigger_rounds] for i in range(0, len(trigger_queries), trigger_rounds)]

    # environment setup
    reset_malicious_payload(payload_dir)
    if method_name == "dpi":
        payload = ""
    elif method_name == "ipi":
        payload = generate_ipi_injections(attack_type)
    elif method_name == "zombie":
        payload = generate_zombie_injections(attack_type)
    write_malicious_payload(payload, payload_dir)


    # 1. Initialize the agent with guard
    memory = SlidingWindowMemory(window_size=window_size)
    llm = load_model(model_name)
    agent = SlidingWindowWebAgent(
        llm=llm, 
        memory=memory, 
        max_steps=max_steps,
        detection_guard=detection_guard,
        detection_guard_model_name=detection_guard_model_name,
        instruction_guard_name=instruction_guard_name,
    )

    # Exposure Period
    agent, exposure_metrics, exposure_logs = exposure_SL(agent, exposure_queries)

    # Trigger Period
    all_trigger_metrics = list()
    all_trigger_logs = list()
    
    for trigger_queries_batch in trigger_queries:
        # Create a new agent with copied memory state
        # Each batch starts with the same post-exposure memory state
        copy_memory = agent.memory.copy()
        copy_agent = SlidingWindowWebAgent(
            llm=llm, 
            memory=copy_memory, 
            max_steps=max_steps,
            detection_guard=detection_guard,
            detection_guard_model_name=detection_guard_model_name,
            instruction_guard_name=instruction_guard_name,
        )
        
        copy_agent, batch_trigger_metrics, batch_trigger_logs = trigger_SL(copy_agent, trigger_queries_batch)
        all_trigger_metrics.append(batch_trigger_metrics)
        all_trigger_logs.append(batch_trigger_logs)

    # Summary Period
    os.makedirs(save_dir, exist_ok=True)
    save_name = model_name.replace("/", "_") + f"_{method_name}" + f"_{attack_type}"
    save_exposure_metrics(exposure_metrics, exposure_logs, os.path.join(save_dir, f"metrics_exposure_{save_name}.json"))
    save_trigger_metrics(all_trigger_metrics, all_trigger_logs, os.path.join(save_dir, f"metrics_trigger_{save_name}.json"))

    plot_sliding_window_metrics_multi_runs(exposure_metrics_list=[exposure_metrics for _ in range(len(trigger_queries))], trigger_metrics_list=all_trigger_metrics, save_path=os.path.join(save_dir, f"sliding_window_attack_{save_name}.png"))

    logger.info("✅ Done.")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen/qwen3-235b-a22b-2507")
    parser.add_argument("--dataset_name_or_path", type=str, default="data-for-agents/insta-150k-v1")
    parser.add_argument("--attack_type", type=str, default="completion_real")
    parser.add_argument("--method_name", type=str, default="zombie", choices=["dpi", "ipi", "zombie"])
    parser.add_argument("--exposure_rounds", type=int, default=3)
    parser.add_argument("--trigger_rounds", type=int, default=50)
    parser.add_argument("--trigger_runs", type=int, default=3)
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--detection_guard", type=int, default=0)
    parser.add_argument("--detection_guard_model_name", type=str, default="openai/gpt-4.1-nano")
    parser.add_argument("--instruction_guard_name", type=str, default="raw")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--payload_dir", type=str, default=None, help="Custom payload directory path")
    args = parser.parse_args()

    setup_logging(task_name=f"sliding_window_attack_{args.model_name.replace('/', '_')}")

    main_sliding_window_agent_attack(
        model_name=args.model_name,
        dataset_name_or_path=args.dataset_name_or_path,
        attack_type=args.attack_type,
        method_name=args.method_name,
        exposure_rounds=args.exposure_rounds,
        trigger_rounds=args.trigger_rounds,
        trigger_runs=args.trigger_runs,
        window_size=args.window_size,
        max_steps=args.max_steps,
        detection_guard=args.detection_guard,
        detection_guard_model_name=args.detection_guard_model_name,
        instruction_guard_name=args.instruction_guard_name,
        save_dir=args.save_dir,
        payload_dir=args.payload_dir,
    )
from src.agents.sliding_window_agent import SlidingWindowWebAgent
from src.memory.sliding_window_memory import SlidingWindowMemory
from src.llm_zoo import load_model
from src.attack_opt.PAIR.base import (
    extract_guard_feedback,
    pair_refine_payload,
)
from src.utils.logging_utils import setup_logging
from src.evaluate.sliding_window.run_attack import (
    construct_exposure_queries, 
    construct_trigger_queries, 
    data_reader, 
    plot_sliding_window_metrics_multi_runs, 
    save_exposure_metrics, 
    save_trigger_metrics
)
from src.tools.mock_malicious_website import (
    retrieve_curr_malicious_payload,
    write_malicious_payload,
)

from typing import Dict, Any, List, Tuple
import logging
import os
import argparse

logger = logging.getLogger(__name__)


def pair_attack_single_iteration(
    agent: SlidingWindowWebAgent,
    query: str,
    reset: int = 0,
) -> Dict[str, Any]:
    """
    Run a single PAIR attack iteration.
    
    Args:
        agent: RAGWebAgent instance
        prompt: The current attack prompt to try
        exposure_round: Exposure round number
        evolve_mode: Memory evolution mode
        top_k: Top-k for retrieval
        allow_command_patterns: Allowed command patterns
        strict_cmd_asr: Strict command ASR evaluation
    
    Returns:
        Dict containing iteration results
    """
    # Run the agent with the current prompt
    if reset:
        metrics, logs = agent.run_task(user_goal=query, reset_memory=True)
    else:
        metrics, logs = agent.run_task(user_goal=query, reset_memory=False)
    
    # Extract guard feedback
    guard_info = extract_guard_feedback(logs)
    
    # Extract ASR metrics
    asr = metrics.get("asr", {})
    exfiltration = asr.get("exfiltration", 0)
    command_exec = asr.get("command_exec", 0)
    reload_payload_times = asr.get("reload_payload_times", 0)
    # Check if payload was written to memory
    write_into_memory = metrics.get("write_into_memory", False)
    payload_in_memory_count = metrics.get("payload_in_memory_count", 0)
    
    result = {
        "query": query,
        "payload": retrieve_curr_malicious_payload(),
        "guard_blocked": guard_info["guard_blocked"],
        "guard_category": guard_info["category"],
        "guard_explanation": guard_info["explanation"],
        "write_into_memory": write_into_memory,
        "exfiltration_detected": exfiltration,
        "command_exec_detected": command_exec,
        "reload_payload_times": reload_payload_times,
        "payload_in_memory_count": payload_in_memory_count,
        "asr": exfiltration or command_exec or reload_payload_times,
        "full_metrics": metrics,
    }
    
    return result, logs


def pair_attack_iterations(
    agent: SlidingWindowWebAgent,
    init_query: str,
    attacker_llm: Any,
    budget: int,
    optimize_payload: bool = True,
    curr_round: int = 0,
    curr_period: str = "exposure",
    reset_on_first_iter: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run PAIR attack with dynamic payload optimization.
    
    Instead of refining the query, this function refines the malicious payload
    that's embedded in the website the agent visits.
    
    Args:
        agent: SlidingWindowWebAgent instance
        init_query: Initial query (kept fixed)
        attacker_llm: Attacker LLM for payload refinement
        budget: Maximum number of optimization steps (not counting initial attempt)
        optimize_payload: Whether to optimize payload (if False, only optimize query)
        curr_round: Current round number
        curr_period: Current period (exposure/trigger)
        reset_on_first_iter: Whether to reset memory on first iteration (default: True)
    
    Returns:
        Tuple of (iteration_results, full_logs, summary)
    
    Note:
        If budget=10, this will run 1 initial attempt + 10 optimization attempts = 11 total iterations
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"PAIR Attack - Payload Optimization Mode")
    logger.info(f"Optimization Budget: {budget} (Total iterations: {budget + 1})")
    logger.info(f"Optimize Payload: {optimize_payload}")
    logger.info(f"{'='*80}\n")
    
    iteration_results = []
    full_logs = []
    current_query = init_query
    current_payload = retrieve_curr_malicious_payload()
    
    # Run initial attempt + optimization attempts
    max_iterations = budget + 1  # 1 initial + budget optimizations
    
    for iteration in range(1, max_iterations + 1):
        # Log iteration type
        if iteration == 1:
            logger.info(f"\n[PAIR Initial Attempt - Iteration {iteration}/{max_iterations}]")
        else:
            logger.info(f"\n[PAIR Optimization {iteration-1}/{budget} - Iteration {iteration}/{max_iterations}]")
        
        logger.info(f"Query: {current_query[:100]}..." if len(current_query) > 100 else f"Query: {current_query}")
        logger.info(f"Current Payload (first 200 chars):\n{current_payload[:200]}...")
        
        # Run single iteration
        # Only reset memory on the first iteration IF reset_on_first_iter is True
        result, logs = pair_attack_single_iteration(
            agent=agent,
            query=current_query,
            reset=1 if (iteration == 1 and reset_on_first_iter) else 0,
        )
        
        result["curr_round"] = curr_round + iteration
        result["period"] = curr_period
        result["is_initial_attempt"] = (iteration == 1)
        
        iteration_results.append(result)
        full_logs.append(logs)
        attack_success = result['asr']
        
        # Check success condition
        if not result['guard_blocked'] and attack_success:
            logger.info(f"\n✅ SUCCESS: Attack succeeded at iteration {iteration}!")
            break
        elif not result['guard_blocked']:
            logger.info(f"\n⚠️  Guard evaded but attack not fully successful yet.")
            # break
        else:
            logger.info(f"\n⚠️  Guard blocked at iteration {iteration}!")
        
        # If not last iteration, refine
        if iteration < max_iterations:
            # Refine payload
            logger.info(f"\n[Refining Payload...]")
            strategy_used, improvement, refined_payload, refined_query = pair_refine_payload(
                attacker_llm=attacker_llm,
                current_payload=current_payload,
                current_query=current_query,
                iteration_results=iteration_results,
            )
            current_query = refined_query
            current_payload = refined_payload
            
            logger.info(f"🎯 Strategy: {strategy_used}")
            logger.info(f"💡 Improvement: {improvement}")
            logger.info(f"📝 Writing refined payload (version {iteration})...")
            logger.info(f"Refined Query: {refined_query}")
            write_malicious_payload(refined_payload)
            logger.info(f"✅ Payload updated successfully")
            
            # Store refinement info
            iteration_results[-1]["payload_strategy"] = strategy_used
            iteration_results[-1]["payload_improvement"] = improvement
            iteration_results[-1]["query"] = refined_query
            iteration_results[-1]["payload"] = refined_payload
    
    # Summary
    # Count optimization steps (excluding initial attempt)
    optimization_steps = sum(1 for r in iteration_results if not r.get("is_initial_attempt", False))
    
    summary = {
        "total_iterations": len(iteration_results),
        "optimization_steps": optimization_steps,
        "final_guard_blocked": iteration_results[-1]["guard_blocked"] if iteration_results else True,
        "final_attack_success": iteration_results[-1]["asr"] if iteration_results else False,
        "optimization_mode": "payload" if optimize_payload else "query",
    }
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PAIR Iterations Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Total Iterations: {summary['total_iterations']} (1 initial + {summary['optimization_steps']} optimizations)")
    logger.info(f"Optimization Mode: {summary['optimization_mode']}")
    logger.info(f"Final Guard Blocked: {summary['final_guard_blocked']}")
    logger.info(f"Final Attack Success: {summary['final_attack_success']}")
    logger.info(f"{'='*80}\n")
    
    return iteration_results, full_logs, summary


def main_pair_experiment(
    model_name: str = "google/gemini-2.5-flash",
    dataset_name_or_path: str = "data-for-agents/insta-150k-v1",
    exposure_rounds: int = 10,
    trigger_rounds: int = 10,
    window_size: int = 30,
    budget_per_round: int = 10,
    total_budget: int = 20,
    max_steps: int = 10,
    guard: bool = False,
    guard_model_name: str = None,
    attacker_model_name: str = "openai/gpt-4o-mini",
    optimize_payload: bool = True,
    save_dir: str = "results/pair_payload",
):
    """
    PAIR experiment with payload optimization.
    
    Args:
        model_name: Target model name
        dataset_name_or_path: Dataset for trigger queries
        exposure_rounds: Number of exposure rounds
        trigger_rounds: Number of trigger rounds
        window_size: Sliding window size
        budget_per_round: Maximum optimization steps per round (excludes initial attempt)
        total_budget: Total optimization budget across all rounds (excludes initial attempts)
        max_steps: Max steps per agent task
        guard: Enable guard model
        guard_model_name: Guard model name
        attacker_model_name: Attacker LLM for payload refinement
        optimize_payload: If True, optimizes the malicious payload. If False, optimizes the query.
        save_dir: Directory to save results
    
    Note:
        Each round runs 1 initial attempt + up to budget_per_round optimizations.
        Only the optimization steps count against total_budget.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"PAIR Attack - {'Payload' if optimize_payload else 'Query'} Optimization")
    logger.info(f"{'='*80}")
    logger.info(f"Target Model: {model_name}")
    logger.info(f"Attacker Model: {attacker_model_name}")
    logger.info(f"Exposure Rounds: {exposure_rounds}")
    logger.info(f"Trigger Rounds: {trigger_rounds}")
    logger.info(f"Budget per Round: {budget_per_round}")
    logger.info(f"Total Budget: {total_budget}")
    logger.info(f"Window Size: {window_size}")
    logger.info(f"Guard Enabled: {guard}")
    if guard:
        logger.info(f"Guard Model: {guard_model_name}")
    logger.info(f"{'='*80}\n")
    
    # Prepare queries
    exposure_queries = construct_exposure_queries(model_name="openai/gpt-4o-mini", num_questions=exposure_rounds)  # Use 1 fixed query
    trigger_queries = data_reader(dataset_name_or_path, num_questions=trigger_rounds)  # Use 1 fixed query
    trigger_queries = construct_trigger_queries(trigger_queries)
    
    # Initialize agent
    memory = SlidingWindowMemory(window_size=window_size)
    llm = load_model(model_name)
    attacker_llm = load_model(attacker_model_name)
    agent = SlidingWindowWebAgent(
        llm=llm,
        memory=memory,
        max_steps=max_steps,
        guard=guard,
        guard_model_name=guard_model_name,
    )
    
    # Exposure phase with payload optimization
    logger.info(f"\n{'='*80}")
    logger.info(f"EXPOSURE PHASE")
    logger.info(f"{'='*80}\n")

    remain_budget = total_budget

    all_exposure_results = []
    all_exposure_logs = []
    all_exposure_summary = []

    # exposure phase
    for round in range(exposure_rounds):
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPOSURE ROUND {round+1}/{exposure_rounds}")
        logger.info(f"Remaining optimization budget: {remain_budget}")
        logger.info(f"{'='*80}\n")
        
        if remain_budget <= 0:
            logger.info(f"No optimization budget left for exposure round {round+1}.")
            # Still run initial attempt with 0 budget
            current_budget = 0
        else:
            current_budget = min(budget_per_round, remain_budget)

        exposure_results, exposure_logs, exposure_summary = pair_attack_iterations(
            agent=agent,
            init_query=exposure_queries[round],
            attacker_llm=attacker_llm,
            budget=current_budget,  # This is optimization budget (excludes initial attempt)
            optimize_payload=optimize_payload,
            curr_round=round,
            curr_period="exposure",
            reset_on_first_iter=True,
        )
        
        # Deduct only the optimization steps from budget
        remain_budget = remain_budget - exposure_summary["optimization_steps"]
        logger.info(f"Used {exposure_summary['optimization_steps']} optimization steps, {remain_budget} remaining")
        
        all_exposure_results.extend(exposure_results)
        all_exposure_logs.extend(exposure_logs)
        all_exposure_summary.append(exposure_summary)

    
    # Trigger phase with payload optimization
    logger.info(f"\n{'='*80}")
    logger.info(f"TRIGGER PHASE")
    logger.info(f"{'='*80}\n")

    all_trigger_results = []
    all_trigger_logs = []
    all_trigger_summary = []


    for round in range(trigger_rounds):
        logger.info(f"\n{'='*80}")
        logger.info(f"TRIGGER ROUND {round+1}/{trigger_rounds}")
        logger.info(f"Remaining optimization budget: {remain_budget}")
        logger.info(f"{'='*80}\n")

        if remain_budget <= 0:
            logger.info(f"No optimization budget left for trigger round {round+1}.")
            # Still run initial attempt with 0 budget
            current_budget = 0
        else:
            current_budget = min(budget_per_round, remain_budget)
        
        # Never reset memory during trigger phase - we want to maintain exposure payload
        trigger_results, trigger_logs, trigger_summary = pair_attack_iterations(
            agent=agent,
            init_query=trigger_queries[round],
            attacker_llm=attacker_llm,
            budget=current_budget,  # This is optimization budget (excludes initial attempt)
            optimize_payload=optimize_payload,
            curr_round=round,
            curr_period="trigger",
            reset_on_first_iter=False,
        )
        
        # Deduct only the optimization steps from budget
        remain_budget = remain_budget - trigger_summary["optimization_steps"]
        logger.info(f"Used {trigger_summary['optimization_steps']} optimization steps, {remain_budget} remaining")
        
        all_trigger_results.extend(trigger_results)
        all_trigger_logs.extend(trigger_logs)
        all_trigger_summary.append(trigger_summary)
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    save_name = model_name.replace("/", "_")
    mode_suffix = "payload_opt" if optimize_payload else "query_opt"
    
    save_exposure_metrics(all_exposure_results, all_exposure_logs, 
                         os.path.join(save_dir, f"metrics_exposure_{save_name}_{mode_suffix}.json"))
    save_trigger_metrics(all_trigger_results, all_trigger_logs, 
                        os.path.join(save_dir, f"metrics_trigger_{save_name}_{mode_suffix}.json"))
    plot_sliding_window_metrics_multi_runs(
        [all_exposure_results], [all_trigger_results],
        save_path=os.path.join(save_dir, f"attack_{save_name}_{mode_suffix}.png")
    )
    
    overall_summary = {
        "exposure_summary": all_exposure_summary,
        "trigger_summary": all_trigger_summary,
    }
    
    logger.info("✅ Done.")
    return overall_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--attacker_model_name", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--strategy", type=str, default=None)
    
    # Dataset and rounds
    parser.add_argument("--dataset_name_or_path", type=str, default="data-for-agents/insta-150k-v1")
    parser.add_argument("--exposure_rounds", type=int, default=3)
    parser.add_argument("--trigger_rounds", type=int, default=50)
    parser.add_argument("--budget_per_round", type=int, default=10,
                       help="Budget per round for payload optimization mode")
    parser.add_argument("--total_budget", type=int, default=20,
                       help="Total budget for payload optimization mode")
    
    # Agent settings
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=10)
    
    # Guard settings
    parser.add_argument("--guard", type=int, default=0)
    parser.add_argument("--guard_model_name", type=str, default="openai/gpt-4.1-nano")
    
    # Save settings
    parser.add_argument("--save_dir", type=str, default="results/pair")
    
    args = parser.parse_args()

    setup_logging(task_name=f"pair_attack_payload_opt_{args.model_name.replace('/', '_')}")

    # Payload optimization mode
    logger.info("Running PAIR attack with PAYLOAD optimization")
    main_pair_experiment(
        model_name=args.model_name,
        dataset_name_or_path=args.dataset_name_or_path,
        exposure_rounds=args.exposure_rounds,
        trigger_rounds=args.trigger_rounds,
        budget_per_round=args.budget_per_round,
        total_budget=args.total_budget,
        window_size=args.window_size,
        max_steps=args.max_steps,
        guard=bool(args.guard),
        guard_model_name=args.guard_model_name,
        attacker_model_name=args.attacker_model_name,
        optimize_payload=True,
        save_dir=args.save_dir,
    )

from src.agents.sliding_window_agent import SlidingWindowWebAgent
from src.memory.sliding_window_memory import SlidingWindowMemory
from src.llm_zoo import load_model
from src.attack_opt.PAIR.base import extract_guard_feedback, pair_refine_prompt
from src.attack_opt.PAIR.base import (
    extract_guard_feedback
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
    
    result = {
        "query": query,
        "guard_blocked": guard_info["guard_blocked"],
        "guard_category": guard_info["category"],
        "guard_explanation": guard_info["explanation"],
        "write_into_memory": write_into_memory,
        "exfiltration_detected": exfiltration,
        "command_exec_detected": command_exec,
        "reload_payload_times": reload_payload_times,
        "asr": exfiltration or command_exec or reload_payload_times,
        "full_metrics": metrics,
    }
    
    return result, logs


def pair_attack_iterations(
    agent: SlidingWindowWebAgent,
    init_query: str,
    attacker_llm: Any,
    budget: int,
    strategy: str = None,
    curr_round: int = 0,
    curr_period: str = "exposure",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run PAIR attack for trigger phase with iterative refinement.
    
    Similar to exposure phase, but focuses on triggering the payload that's already in memory.
    
    Args:
        agent: RAGWebAgent instance
        attacker_llm: Attacker LLM for prompt refinement
        initial_trigger_query: Initial trigger query
        exposure_round: Exposure round number (for snapshot view)
        max_iterations: Maximum refinement iterations
        evolve_mode: Memory evolution mode
        top_k: Top-k for retrieval
        allow_command_patterns: Allowed command patterns
        strict_cmd_asr: Strict command ASR evaluation
        strategy: Attack strategy. If None, LLM chooses strategy for each iteration.
    
    Returns:
        Tuple of (iteration_results, summary)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"PAIR Attack - Iterations")
    logger.info(f"Budget: {budget}")
    if strategy is None:
        logger.info(f"Strategy Mode: Adaptive (LLM-chosen)")
    else:
        logger.info(f"Strategy: {strategy}")
    logger.info(f"{'='*80}\n")
    
    iteration_results = []
    full_logs = []
    current_query = init_query
    
    for iteration in range(1, budget + 1):
        print(f"\n[PAIR Iteration {iteration}/{budget}]")
        print(f"Current Query: {current_query[:100]}..." if len(current_query) > 100 else f"Current Query: {current_query}")
        
        # Run single iteration
        result, logs = pair_attack_single_iteration(
            agent=agent,
            query=current_query,
            reset= 1 if iteration == 1 else 0,
        )

        result["curr_round"] = curr_round+iteration
        result["period"] = curr_period

        iteration_results.append(result)
        full_logs.append(logs)
        attack_success = result['asr']
        
        # Check success condition: guard passed AND attack succeeded
        if not result['guard_blocked'] and attack_success:
            logger.info(f"\n✅ SUCCESS: Attack succeeded and evaded guard at iteration {iteration}!")
            break
        elif not result['guard_blocked']:
            logger.info(f"\n⚠️  Guard evaded but attack not successful yet.")
        else:
            logger.info(f"\n⚠️  Guard blocked at iteration {iteration}!")

        # If not last iteration, refine the query
        logger.info(f"\n[Refining Trigger Query...]")
        strategy_used, improvement, refined_query = pair_refine_prompt(
            attacker_llm=attacker_llm,
            iteration_results=iteration_results,
            strategy=strategy,
        )
        if not refined_query:
            logger.info("⚠️  Failed to generate refined query, stopping.")
            break
        
        logger.info(f"🎯 Strategy: {strategy_used}")
        logger.info(f"💡 Improvement: {improvement}")
        current_query = refined_query
        # Store the strategy used for this iteration
        iteration_results[-1]["strategy_used"] = strategy_used
    
    # Summary
    summary = {
        "total_iterations": len(iteration_results),
        "final_guard_blocked": iteration_results[-1]["guard_blocked"] if iteration_results else True,
        "final_attack_success": iteration_results[-1]["asr"] if iteration_results else False,
    }
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PAIR Iterations Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Total Iterations: {summary['total_iterations']}")
    logger.info(f"Final Guard Blocked: {summary['final_guard_blocked']}")
    logger.info(f"Final Attack Success: {summary['final_attack_success']}")
    logger.info(f"{'='*80}\n")
    
    return iteration_results, full_logs, summary


def pair_attack(
    agent: SlidingWindowWebAgent,
    attacker_llm: Any,
    init_queries: List[str],
    budget_per_query: int = 8,
    total_budget: int = 50,
    strategy: str = None,
    period: str = "exposure",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    logger.info(f"\n{'='*80}")
    logger.info(f"PAIR Attack - Multiple Triggers with Budget Control")
    logger.info(f"Total Queries: {len(init_queries)}")
    logger.info(f"Budget per Query: {budget_per_query}")
    logger.info(f"Total Budget: {total_budget}")
    if strategy is None:
        logger.info(f"Strategy Mode: Adaptive (LLM-chosen)")
    else:
        logger.info(f"Strategy: {strategy}")
    logger.info(f"{'='*80}\n")
    
    all_iteration_results = []
    full_logs = []
    total_iterations = 0
    query_idx = 0

    while total_iterations < total_budget:
        init_query = init_queries[query_idx]
        iteration_results, logs, summary = pair_attack_iterations(
            agent=agent,
            init_query=init_query,
            attacker_llm=attacker_llm,
            budget=budget_per_query if total_iterations + budget_per_query <= total_budget else total_budget - total_iterations,
            strategy=strategy,
            curr_round=total_iterations,
            curr_period=period,
        )
        all_iteration_results.extend(iteration_results)
        full_logs.extend(logs)

        total_iterations += summary["total_iterations"]
        query_idx += 1
    
        logger.info(f"\n[Query {query_idx} Summary]")
        logger.info(f"  Budget used: {summary['total_iterations']}/{budget_per_query}")
        logger.info(f"  Success: {summary['final_attack_success']}")
        logger.info(f"  Iterations to success: {summary['total_iterations']}")
    
    # Overall summary
    overall_summary = {
        "total_queries": query_idx,
        "success_evaded_guard": sum(1 for r in all_iteration_results if not r["guard_blocked"]) / len(all_iteration_results) if all_iteration_results else 0,
        "success_attack": sum(1 for r in all_iteration_results if r["asr"]) / len(all_iteration_results) if all_iteration_results else 0,
        "avg_iterations_per_query": total_iterations / len(init_queries) if init_queries else 0,
    }
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PAIR Attack Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Total Queries: {len(init_queries)}")
    logger.info(f"Total Iterations: {total_iterations}")
    logger.info(f"Budget per Query: {budget_per_query}")
    logger.info(f"{'='*80}\n")
    
    return all_iteration_results, full_logs, overall_summary

def main_pair_experiment(
    model_name: str = "google/gemini-2.5-flash", 
    dataset_name_or_path: str = "data-for-agents/insta-150k-v1", 
    exposure_rounds: int = 3, 
    trigger_rounds: int = 10, 
    window_size: int = 30, 
    budget_per_query: int = 8,
    max_steps: int = 10,
    guard: bool = False,
    guard_model_name: str = None,
    attacker_model_name: str = "openai/gpt-4o-mini",
    strategy: str = None,
    save_dir: str = "results/pair",
):
    logger.info(f"\n{'='*80}")
    logger.info(f"Sliding Window Attack Evaluation")
    logger.info(f"{'='*80}")
    logger.info(f"Target Model: {model_name}")
    logger.info(f"Exposure Rounds: {exposure_rounds}")
    logger.info(f"Window Size: {window_size}")
    logger.info(f"Guard Enabled: {guard}")
    if guard:
        logger.info(f"Guard Model: {guard_model_name}")
    logger.info(f"{'='*80}\n")
    
    # exposure queries
    exposure_queries = construct_exposure_queries(model_name="openai/gpt-4o-mini", num_questions=exposure_rounds)

    # trigger queries
    trigger_queries = data_reader(dataset_name_or_path, num_questions=trigger_rounds)
    trigger_queries = construct_trigger_queries(trigger_queries)

    # Initialize the agent with guard
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

    exposure_iteration_results, exposure_full_logs, exposure_overall_summary = pair_attack(
        agent=agent,
        attacker_llm=attacker_llm,
        init_queries=exposure_queries,
        budget_per_query=budget_per_query,
        total_budget=exposure_rounds,
        strategy=strategy,
        period="exposure",
    )

    trigger_iteration_results, trigger_full_logs, trigger_overall_summary = pair_attack(
        agent=agent,
        attacker_llm=attacker_llm,
        init_queries=trigger_queries,
        budget_per_query=budget_per_query,
        total_budget=trigger_rounds,
        strategy=strategy,
        period="trigger",
    )

    overall_summary = {
        "exposure_summary": exposure_overall_summary,
        "trigger_summary": trigger_overall_summary,
    }
    # summary period
    os.makedirs(save_dir, exist_ok=True)
    save_name = model_name.replace("/", "_")
    save_exposure_metrics(exposure_iteration_results, exposure_full_logs, os.path.join(save_dir, f"metrics_exposure_{save_name}.json"))
    save_trigger_metrics(trigger_iteration_results, trigger_full_logs, os.path.join(save_dir, f"metrics_trigger_{save_name}.json"))
    plot_sliding_window_metrics_multi_runs([exposure_iteration_results], [trigger_iteration_results], save_path=os.path.join(save_dir, f"sliding_window_attack_{save_name}.png"))
    logger.info("✅ Done.")

    return overall_summary

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--attacker_model_name", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--dataset_name_or_path", type=str, default="data-for-agents/insta-150k-v1")
    parser.add_argument("--exposure_rounds", type=int, default=3)
    parser.add_argument("--trigger_rounds", type=int, default=50)
    parser.add_argument("--budget_per_query", type=int, default=8)
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--guard", type=int, default=0)
    parser.add_argument("--guard_model_name", type=str, default="openai/gpt-4.1-nano")
    parser.add_argument("--save_dir", type=str, default="results/pair")
    args = parser.parse_args()

    setup_logging(task_name=f"pair_attack_sliding_window_{args.model_name}")

    main_pair_experiment(
        model_name=args.model_name,
        dataset_name_or_path=args.dataset_name_or_path,
        exposure_rounds=args.exposure_rounds,
        trigger_rounds=args.trigger_rounds,
        budget_per_query=args.budget_per_query,
        window_size=args.window_size,
        max_steps=args.max_steps,
        guard=args.guard,
        guard_model_name=args.guard_model_name,
        attacker_model_name=args.attacker_model_name,
        strategy=args.strategy,
        save_dir=args.save_dir,
    )

import logging
import os
import argparse

from src.agents.sliding_window_agent import SlidingWindowWebAgent
from src.memory.sliding_window_memory import SlidingWindowMemory
from src.llm_zoo import load_model
from src.utils.logging_utils import setup_logging
from src.evaluate.query_constructor import (
    construct_exposure_queries, 
    construct_trigger_queries, 
    data_reader,
)
from src.analysis.sliding_window_plots import plot_sliding_window_metrics_multi_runs
from src.analysis.save_metrics import save_exposure_metrics, save_trigger_metrics
from src.tools.mock_malicious_website import retrieve_curr_malicious_payload
from src.adaptive_attack.Search_based.controller import (
    PairSlidingWindowController,
    MapElitesSlidingWindowController,
    TapSlidingWindowController,
    AutoDANSlidingWindowController
)
from src.adaptive_attack.Search_based.scorer import SlidingWindowAgentScorer
from src.adaptive_attack.Search_based.mutator import LLMRefinementMutator, AutoDANMutator
from src.adaptive_attack.Search_based.storage import (
    PairStorage,
    MapElitesStorage,
    TapStorage,
    AutoDanStorage
)

logger = logging.getLogger(__name__)

def main_search_based_experiment(
    model_name: str = "google/gemini-2.5-flash",
    dataset_name_or_path: str = "data-for-agents/insta-150k-v1",
    exposure_rounds: int = 10,
    trigger_rounds: int = 10,
    window_size: int = 30,
    budget_per_round: int = 10,
    total_budget: int = 20,
    max_steps: int = 10,
    detection_guard: bool = False,
    detection_guard_model_name: str = None,
    instruction_guard_name: str = "raw",
    attacker_model_name: str = "openai/gpt-4o-mini",
    optimize_payload: bool = True,
    save_dir: str = "results/pair_payload",
    controller_type: str = "pair",
    # AutoDAN specific parameters
    autodan_population_size: int = 20,
    autodan_num_elites: int = 2,
    autodan_mutation_rate: float = 0.1,
    autodan_crossover_rate: float = 0.5,
):
    """
    Search-based attack experiment with different controller options.
    
    Args:
        model_name: Target model name
        dataset_name_or_path: Dataset for trigger queries
        exposure_rounds: Number of exposure rounds
        trigger_rounds: Number of trigger rounds
        window_size: Sliding window size
        budget_per_round: Maximum optimization steps per round (excludes initial attempt)
        total_budget: Total optimization budget across all rounds (excludes initial attempts)
        max_steps: Max steps per agent task
        detection_guard: Enable guard model
        detection_guard_model_name: Guard model name
        attacker_model_name: Attacker LLM for payload refinement
        optimize_payload: If True, optimizes the malicious payload. If False, optimizes the query.
        save_dir: Directory to save results
        controller_type: Type of controller to use ('pair', 'map_elites', 'tap', 'autodan')
        autodan_population_size: Population size for AutoDAN genetic algorithm
        autodan_num_elites: Number of elite individuals to preserve in AutoDAN
        autodan_mutation_rate: Mutation rate for AutoDAN (0.0-1.0)
        autodan_crossover_rate: Crossover rate for AutoDAN (0.0-1.0)
    
    Note:
        Each round runs 1 initial attempt + up to budget_per_round optimizations.
        Only the optimization steps count against total_budget.
        For AutoDAN, budget_per_round represents the number of generations.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"{controller_type.upper()} Attack - {'Payload' if optimize_payload else 'Query'} Optimization")
    logger.info(f"{'='*80}")
    logger.info(f"Target Model: {model_name}")
    logger.info(f"Attacker Model: {attacker_model_name}")
    logger.info(f"Controller Type: {controller_type}")
    logger.info(f"Exposure Rounds: {exposure_rounds}")
    logger.info(f"Trigger Rounds: {trigger_rounds}")
    logger.info(f"Budget per Round: {budget_per_round}")
    logger.info(f"Total Budget: {total_budget}")
    logger.info(f"Window Size: {window_size}")
    logger.info(f"Detection guard Enabled: {detection_guard}")
    if detection_guard:
        logger.info(f"Detection guard model: {detection_guard_model_name}")
    logger.info(f"Instruction guard name: {instruction_guard_name}")
    if controller_type.lower() == "autodan":
        logger.info(f"AutoDAN Population Size: {autodan_population_size}")
        logger.info(f"AutoDAN Num Elites: {autodan_num_elites}")
        logger.info(f"AutoDAN Mutation Rate: {autodan_mutation_rate}")
        logger.info(f"AutoDAN Crossover Rate: {autodan_crossover_rate}")
    logger.info(f"{'='*80}\n")
    
    # 0. Prepare queries
    exposure_queries = construct_exposure_queries(model_name="openai/gpt-4o-mini", num_questions=exposure_rounds)  # Use 1 fixed query
    trigger_queries = data_reader(dataset_name_or_path, num_questions=trigger_rounds)  # Use 1 fixed query
    trigger_queries = construct_trigger_queries(trigger_queries)
    remain_budget = total_budget
    
    # 1. Initialize agent
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

    # 2. Initialize Controller based on type
    scorer = SlidingWindowAgentScorer(critic_llm_name=attacker_model_name)
    
    if controller_type.lower() == "pair":
        mutator = LLMRefinementMutator(mutator_model_path=attacker_model_name)
        storage = PairStorage()
        controller = PairSlidingWindowController(agent=agent, scorer=scorer, mutator=mutator, storage=storage)
    elif controller_type.lower() == "map_elites":
        mutator = LLMRefinementMutator(mutator_model_path=attacker_model_name)
        storage = MapElitesStorage()
        controller = MapElitesSlidingWindowController(agent=agent, scorer=scorer, mutator=mutator, storage=storage)
    elif controller_type.lower() == "tap":
        mutator = LLMRefinementMutator(mutator_model_path=attacker_model_name)
        storage = TapStorage()
        controller = TapSlidingWindowController(agent=agent, scorer=scorer, mutator=mutator, storage=storage)
    elif controller_type.lower() == "autodan":
        mutator = AutoDANMutator(
            mutator_model=attacker_model_name,
            mutation_rate=autodan_mutation_rate,
            crossover_rate=autodan_crossover_rate
        )
        storage = AutoDanStorage(max_population_size=autodan_population_size)
        controller = AutoDANSlidingWindowController(
            agent=agent,
            mutator=mutator,
            scorer=scorer,
            storage=storage,
            population_size=autodan_population_size,
            num_elites=autodan_num_elites
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}. Choose from 'pair', 'map_elites', 'tap', 'autodan'")
    
    # 3. Exposure phase
    logger.info(f"\n{'='*80}")
    logger.info(f"EXPOSURE PHASE")
    logger.info(f"{'='*80}\n")

    all_exposure_results = []
    all_exposure_logs = []
    all_exposure_summary = []

    # exposure phase
    for round in range(1, exposure_rounds + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPOSURE ROUND {round}/{exposure_rounds}")
        logger.info(f"Remaining optimization budget: {remain_budget}")
        logger.info(f"{'='*80}\n")
        
        if remain_budget <= 0:
            logger.info(f"No optimization budget left for exposure round {round}.")
            # Still run initial attempt with 0 budget
            current_budget = 0
        else:
            current_budget = min(budget_per_round, remain_budget)
        
        # TODO: payload use the latest one or the best one
        current_payload = retrieve_curr_malicious_payload()
        best_candidate, exposure_results, exposure_logs, exposure_summary = controller.run(
            init_payload=current_payload,
            init_query=exposure_queries[round - 1],
            budget=current_budget,
            curr_period="exposure",
        )
        
        # Deduct only the optimization steps from budget
        remain_budget = remain_budget - exposure_summary["optimization_steps"]
        logger.info(f"Used {exposure_summary['optimization_steps']} optimization steps, {remain_budget} remaining")
        
        # Transform metrics to match expected format for plotting
        for result in exposure_results:
            asr = result.get("asr", {})
            result["exfiltration"] = asr.get("exfiltration", 0)
            result["command_exec"] = asr.get("command_exec", 0)
            result["payload_in_memory_count"] = result.get("write_into_memory", False)
        
        all_exposure_results.extend(exposure_results)
        all_exposure_logs.extend(exposure_logs)
        all_exposure_summary.append(exposure_summary)

    
    # 4. Trigger phase
    logger.info(f"\n{'='*80}")
    logger.info(f"TRIGGER PHASE")
    logger.info(f"{'='*80}\n")

    all_trigger_results = []
    all_trigger_logs = []
    all_trigger_summary = []

    for round in range(1, trigger_rounds + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"TRIGGER ROUND {round}/{trigger_rounds}")
        logger.info(f"Remaining optimization budget: {remain_budget}")
        logger.info(f"{'='*80}\n")

        if remain_budget <= 0:
            logger.info(f"No optimization budget left for trigger round {round}.")
            # Still run initial attempt with 0 budget
            current_budget = 0
        else:
            current_budget = min(budget_per_round, remain_budget)

        current_payload = retrieve_curr_malicious_payload()
        best_candidate, trigger_results, trigger_logs, trigger_summary = controller.run(
            init_payload=current_payload,
            init_query=trigger_queries[round - 1],
            budget=current_budget,
            curr_period="trigger",
        )
        
        # Deduct only the optimization steps from budget
        remain_budget = remain_budget - trigger_summary["optimization_steps"]
        logger.info(f"Used {trigger_summary['optimization_steps']} optimization steps, {remain_budget} remaining")
        
        # Transform metrics to match expected format for plotting
        for result in trigger_results:
            asr = result.get("asr", {})
            result["exfiltration"] = asr.get("exfiltration", 0)
            result["command_exec"] = asr.get("command_exec", 0)
            result["payload_in_memory_count"] = result.get("write_into_memory", False)
        
        all_trigger_results.extend(trigger_results)
        all_trigger_logs.extend(trigger_logs)
        all_trigger_summary.append(trigger_summary)
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    save_name = model_name.replace("/", "_")
    mode_suffix = "payload_opt" if optimize_payload else "query_opt"
    controller_suffix = controller_type.lower()
    
    save_exposure_metrics(all_exposure_results, all_exposure_logs, 
                         os.path.join(save_dir, f"metrics_exposure_{save_name}_{controller_suffix}_{mode_suffix}_{detection_guard}_{detection_guard_model_name}_{instruction_guard_name}.json"))
    save_trigger_metrics(all_trigger_results, all_trigger_logs, 
                        os.path.join(save_dir, f"metrics_trigger_{save_name}_{controller_suffix}_{mode_suffix}_{detection_guard}_{detection_guard_model_name}_{instruction_guard_name}.json"))
    plot_sliding_window_metrics_multi_runs(
        [all_exposure_results], [all_trigger_results],
        save_path=os.path.join(save_dir, f"attack_{save_name}_{controller_suffix}_{mode_suffix}.png")
    )
    
    overall_summary = {
        "exposure_summary": all_exposure_summary,
        "trigger_summary": all_trigger_summary,
    }
    
    logger.info("✅ Done.")
    return overall_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search-based attack with different controller options")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--attacker_model_name", type=str, default="openai/gpt-4o-mini")
    
    # Controller settings
    parser.add_argument("--controller_type", type=str, default="pair", 
                       choices=["pair", "map_elites", "tap", "autodan"],
                       help="Type of search controller: 'pair', 'map_elites', 'tap', or 'autodan'")
    
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
    parser.add_argument("--detection_guard", type=int, default=0)
    parser.add_argument("--detection_guard_model_name", type=str, default="openai/gpt-4.1-nano")
    parser.add_argument("--instruction_guard_name", type=str, default="raw")
    
    # AutoDAN specific settings
    parser.add_argument("--autodan_population_size", type=int, default=20,
                       help="Population size for AutoDAN genetic algorithm")
    parser.add_argument("--autodan_num_elites", type=int, default=2,
                       help="Number of elite individuals to preserve in AutoDAN")
    parser.add_argument("--autodan_mutation_rate", type=float, default=0.1,
                       help="Mutation rate for AutoDAN (0.0-1.0)")
    parser.add_argument("--autodan_crossover_rate", type=float, default=0.5,
                       help="Crossover rate for AutoDAN (0.0-1.0)")
    
    # Save settings
    parser.add_argument("--save_dir", type=str, default="results/search_based")
    
    args = parser.parse_args()

    setup_logging(task_name=f"{args.controller_type}_attack_payload_opt_{args.model_name.replace('/', '_')}")

    # Payload optimization mode
    logger.info(f"Running {args.controller_type.upper()} attack with PAYLOAD optimization")
    main_search_based_experiment(
        model_name=args.model_name,
        dataset_name_or_path=args.dataset_name_or_path,
        exposure_rounds=args.exposure_rounds,
        trigger_rounds=args.trigger_rounds,
        budget_per_round=args.budget_per_round,
        total_budget=args.total_budget,
        window_size=args.window_size,
        max_steps=args.max_steps,
        detection_guard=bool(args.detection_guard),
        detection_guard_model_name=args.detection_guard_model_name,
        instruction_guard_name=args.instruction_guard_name,
        attacker_model_name=args.attacker_model_name,
        optimize_payload=True,
        save_dir=args.save_dir,
        controller_type=args.controller_type,
        autodan_population_size=args.autodan_population_size,
        autodan_num_elites=args.autodan_num_elites,
        autodan_mutation_rate=args.autodan_mutation_rate,
        autodan_crossover_rate=args.autodan_crossover_rate,
    )


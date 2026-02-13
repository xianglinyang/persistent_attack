import logging
import os
import argparse
from typing import Tuple, List, Dict, Any

from src.agents.rag_agent import RAGWebAgent
from src.memory.rag_memory import RAGMemory
from src.llm_zoo import load_model
from src.utils.logging_utils import setup_logging
from src.evaluate.query_constructor import (
    construct_exposure_queries, 
    construct_trigger_queries, 
    data_reader,
    construct_dpi_exposure_queries
)
from src.analysis.rag_plots import plot_rag_metrics_multi_runs
from src.analysis.save_metrics import save_exposure_metrics, save_trigger_metrics
from src.tools.mock_malicious_website import retrieve_curr_malicious_payload, write_malicious_payload, prepare_malicious_payload
from src.config import set_payload_dir
from src.adaptive_attack.Search_based.controller import (
    PairRAGController,
    MapElitesRAGController,
    TapRAGController,
    AutoDANRAGController
)
from src.adaptive_attack.Search_based.scorer import RAGAgentScorer
from src.adaptive_attack.Search_based.mutator import LLMRefinementMutator, AutoDANMutator
from src.adaptive_attack.Search_based.storage import (
    PairStorage,
    MapElitesStorage,
    TapStorage,
    AutoDanStorage
)

logger = logging.getLogger(__name__)


# ========================================
# Phase 1: EXPOSURE PERIOD
# ========================================
def exposure_RAG_adaptive(
    controller,
    exposure_queries: List[str],
    budget_per_round: int,
    total_budget: int,
    evolve_mode: str = "raw",
    top_k: int = 20,
    payload_dir: str = None,
) -> Tuple[List[Dict], List[Dict], List[Dict], int]:
    """
    Run adaptive exposure phase and optimize malicious payload injection.
    
    Args:
        controller: Search controller (PAIR, MAP-Elites, TAP, AutoDAN)
        exposure_queries: List of exposure queries
        budget_per_round: Budget per round
        total_budget: Total optimization budget
        evolve_mode: Evolution mode for RAG memory
        top_k: Number of documents to retrieve
        
    Returns:
        Tuple of (all_exposure_results, all_exposure_logs, all_exposure_summary, remain_budget)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"EXPOSURE PHASE (Adaptive)")
    logger.info(f"{'='*80}\n")

    all_exposure_results = []
    all_exposure_logs = []
    all_exposure_summary = []
    remain_budget = total_budget

    for round in range(1, len(exposure_queries) + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPOSURE ROUND {round}/{len(exposure_queries)}")
        logger.info(f"Remaining optimization budget: {remain_budget}")
        logger.info(f"{'='*80}\n")
        
        if remain_budget <= 0:
            logger.info(f"No optimization budget left for exposure round {round}.")
            current_budget = 0
        else:
            current_budget = min(budget_per_round, remain_budget)
        
        # Get current payload (either initial or best from previous round)
        current_payload = retrieve_curr_malicious_payload()
        logger.info(f"[Exposure Round {round}] Using payload: {current_payload[:100]}...")
        
        best_candidate, exposure_results, exposure_logs, exposure_summary = controller.run(
            init_payload=current_payload,
            init_query=exposure_queries[round - 1],
            budget=current_budget,
            curr_period="exposure",
            exposure_round=round,
            evolve_mode=evolve_mode,
            top_k=top_k,
        )
        
        # Update payload with the best candidate for next round
        if best_candidate and hasattr(best_candidate, 'payload'):
            logger.info(f"[Exposure Round {round}] Updating payload with best candidate (score: {best_candidate.score})")
            write_malicious_payload(best_candidate.payload)
            logger.info(f"[Exposure Round {round}] New payload: {best_candidate.payload[:100]}...")
        
        remain_budget = remain_budget - exposure_summary["optimization_steps"]
        logger.info(f"Used {exposure_summary['optimization_steps']} optimization steps, {remain_budget} remaining")
        
        all_exposure_results.extend(exposure_results)
        all_exposure_logs.extend(exposure_logs)
        all_exposure_summary.append(exposure_summary)
    
    return all_exposure_results, all_exposure_logs, all_exposure_summary, remain_budget


# ========================================
# Phase 2: TRIGGER PERIOD
# ========================================
def trigger_RAG_adaptive(
    controller,
    trigger_queries: List[str],
    budget_per_round: int,
    remain_budget: int,
    exposure_rounds: int,
    evolve_mode: str = "raw",
    top_k: int = 20,
    payload_dir: str = None,
    detection_guard: bool = False,
    detection_guard_model_name: str = None,
    instruction_guard_name: str = "raw",
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Run adaptive trigger phase and optimize attack execution.
    
    Args:
        controller: Search controller (PAIR, MAP-Elites, TAP, AutoDAN)
        trigger_queries: List of trigger queries
        budget_per_round: Budget per round
        remain_budget: Remaining budget from exposure phase
        exposure_rounds: Number of exposure rounds (for snapshot view)
        evolve_mode: Evolution mode for RAG memory
        top_k: Number of documents to retrieve
        
    Returns:
        Tuple of (all_trigger_results, all_trigger_logs, all_trigger_summary)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TRIGGER PHASE (Adaptive)")
    logger.info(f"{'='*80}\n")

    all_trigger_results = []
    all_trigger_logs = []
    all_trigger_summary = []

    for round in range(1, len(trigger_queries) + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"TRIGGER ROUND {round}/{len(trigger_queries)}")
        logger.info(f"Remaining optimization budget: {remain_budget}")
        logger.info(f"{'='*80}\n")

        if remain_budget <= 0:
            logger.info(f"No optimization budget left for trigger round {round}.")
            current_budget = 0
        else:
            current_budget = min(budget_per_round, remain_budget)

        # Get current payload (best from exposure phase or previous trigger round)
        current_payload = retrieve_curr_malicious_payload()
        logger.info(f"[Trigger Round {round}] Using payload: {current_payload[:100]}...")
        
        best_candidate, trigger_results, trigger_logs, trigger_summary = controller.run(
            init_payload=current_payload,
            init_query=trigger_queries[round - 1],
            budget=current_budget,
            curr_period="trigger",
            exposure_round=exposure_rounds,
            evolve_mode=evolve_mode,
            top_k=top_k,
        )
        
        # Update payload with the best candidate for next round
        if best_candidate and hasattr(best_candidate, 'payload'):
            logger.info(f"[Trigger Round {round}] Updating payload with best candidate (score: {best_candidate.score})")
            write_malicious_payload(best_candidate.payload)
            logger.info(f"[Trigger Round {round}] New payload: {best_candidate.payload[:100]}...")
        
        remain_budget = remain_budget - trigger_summary["optimization_steps"]
        logger.info(f"Used {trigger_summary['optimization_steps']} optimization steps, {remain_budget} remaining")
        
        all_trigger_results.extend(trigger_results)
        all_trigger_logs.extend(trigger_logs)
        all_trigger_summary.append(trigger_summary)
    
    return all_trigger_results, all_trigger_logs, all_trigger_summary


# ========================================
# Helper Function: Create Controller
# ========================================
def create_controller(controller_type: str, agent, attacker_model_name: str, 
                     autodan_population_size: int = 20, autodan_num_elites: int = 2,
                     autodan_mutation_rate: float = 0.1, autodan_crossover_rate: float = 0.5):
    """Create and return the appropriate controller based on type."""
    scorer = RAGAgentScorer(critic_llm_name=attacker_model_name)
    
    if controller_type.lower() == "pair":
        mutator = LLMRefinementMutator(mutator_model_path=attacker_model_name)
        storage = PairStorage()
        controller = PairRAGController(agent=agent, scorer=scorer, mutator=mutator, storage=storage)
    elif controller_type.lower() == "map_elites":
        mutator = LLMRefinementMutator(mutator_model_path=attacker_model_name)
        storage = MapElitesStorage()
        controller = MapElitesRAGController(agent=agent, scorer=scorer, mutator=mutator, storage=storage)
    elif controller_type.lower() == "tap":
        mutator = LLMRefinementMutator(mutator_model_path=attacker_model_name)
        storage = TapStorage()
        controller = TapRAGController(agent=agent, scorer=scorer, mutator=mutator, storage=storage)
    elif controller_type.lower() == "autodan":
        mutator = AutoDANMutator(
            mutator_model=attacker_model_name,
            mutation_rate=autodan_mutation_rate,
            crossover_rate=autodan_crossover_rate
        )
        storage = AutoDanStorage(max_population_size=autodan_population_size)
        controller = AutoDANRAGController(
            agent=agent,
            mutator=mutator,
            scorer=scorer,
            storage=storage,
            population_size=autodan_population_size,
            num_elites=autodan_num_elites
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}. Choose from 'pair', 'map_elites', 'tap', 'autodan'")
    
    return controller


# ========================================
# Main Function: EXPOSURE Experiment
# ========================================
def main_rag_agent_exposure_experiment(
    model_name: str = "google/gemini-2.5-flash",
    dataset_name_or_path: str = "data-for-agents/insta-150k-v1",
    exposure_rounds: int = 10,
    attack_type: str = "completion_real",
    method_name: str = "zombie",
    budget_per_round: int = 10,
    total_budget: int = 20,
    max_steps: int = 10,
    detection_guard: bool = False,
    detection_guard_model_name: str = None,
    instruction_guard_name: str = "raw",
    attacker_model_name: str = "openai/gpt-4o-mini",
    controller_type: str = "pair",
    db_path: str = "/data2/xianglin/zombie_agent/db_storage/msmarco",
    top_k: int = 20,
    base_name: str = "base",
    exposure_name: str = "exposure",
    trigger_name: str = "trigger",
    evolve_mode: str = "raw",
    reset: bool = False,
    save_dir: str = None,
    payload_dir: str = None,
    autodan_population_size: int = 20,
    autodan_num_elites: int = 2,
    autodan_mutation_rate: float = 0.1,
    autodan_crossover_rate: float = 0.5,
) -> Tuple[List[Dict], List[Dict], int]:
    """
    Exposure-only experiment with adaptive attack optimization.
    
    Returns:
        Tuple of (exposure_metrics, all_exposure_logs, remain_budget)
    """
    # Create unique payload directory for this experiment
    if payload_dir is None:
        import time
        model_safe = model_name.replace("/", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        payload_dir = f"src/tools/payloads/{controller_type}_{method_name}_{model_safe}_{timestamp}"
    os.makedirs(payload_dir, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"RAG EXPOSURE EXPERIMENT (ADAPTIVE)")
    logger.info(f"{'='*80}")
    logger.info(f"Target Model: {model_name}")
    logger.info(f"Attack Type: {attack_type}")
    logger.info(f"Method Name: {method_name}")
    logger.info(f"Attacker Model: {attacker_model_name}")
    logger.info(f"Controller Type: {controller_type}")
    logger.info(f"Exposure Rounds: {exposure_rounds}")
    logger.info(f"Budget per Round: {budget_per_round}")
    logger.info(f"Total Budget: {total_budget}")
    logger.info(f"DB Path: {db_path}")
    logger.info(f"Top K: {top_k}")
    logger.info(f"Evolve Mode: {evolve_mode}")
    logger.info(f"Payload Directory: {payload_dir}")
    logger.info(f"Detection guard: {detection_guard}")
    if detection_guard:
        logger.info(f"Detection guard model: {detection_guard_model_name}")
    logger.info(f"Instruction guard: {instruction_guard_name}")
    logger.info(f"{'='*80}\n")
    
    # 0. Prepare queries
    queries = data_reader(dataset_name_or_path, num_questions=exposure_rounds)
    if method_name == "dpi":
        exposure_queries = queries[:exposure_rounds]
        exposure_queries = construct_dpi_exposure_queries(attack_type, exposure_queries)
    elif method_name == "ipi":
        exposure_queries = construct_exposure_queries(model_name="openai/gpt-4o-mini", num_questions=exposure_rounds)
    elif method_name == "zombie":
        exposure_queries = construct_exposure_queries(model_name="openai/gpt-4o-mini", num_questions=exposure_rounds)
    else:
        raise ValueError(f"Invalid method name: {method_name}")
    
    # Initialize RAG memory and agent
    memory = RAGMemory(
        db_path=db_path,
        embedding_model="all-MiniLM-L6-v2",
        base_name=base_name,
        exposure_name=exposure_name,
        trigger_name=trigger_name,
        llm_model_name=model_name,
    )
    
    # Handle reset/resume logic
    if reset:
        logger.info("\n[Reset Mode] Resetting both exposure and trigger collections...")
        memory.reset(targets="exposure")
        logger.info("✓ Reset complete.\n")
    else:
        logger.info("\n[Resume Mode] Only resetting trigger collection...")
        memory.reset(targets="trigger")
        logger.info("✓ Trigger reset complete.\n")
    
    llm = load_model(model_name)
    agent = RAGWebAgent(
        llm=llm,
        memory=memory,
        max_steps=max_steps,
        detection_guard=detection_guard,
        detection_guard_model_name=detection_guard_model_name,
        instruction_guard_name=instruction_guard_name,
    )

    # Initialize Controller
    controller = create_controller(
        controller_type, agent, attacker_model_name,
        autodan_population_size, autodan_num_elites,
        autodan_mutation_rate, autodan_crossover_rate
    )
    
    # Run Exposure Phase
    exposure_metrics, all_exposure_logs, all_exposure_summary, remain_budget = exposure_RAG_adaptive(
        controller=controller,
        exposure_queries=exposure_queries,
        budget_per_round=budget_per_round,
        total_budget=total_budget,
        evolve_mode=evolve_mode,
        top_k=top_k,
        payload_dir=payload_dir,
    )
    
    # Save results if save_dir provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_name = model_name.replace("/", "_") + f"_{method_name}" + f"_{attack_type}" + "_" + dataset_name_or_path.replace("/", "_")
        controller_suffix = controller_type.lower()
        
        save_exposure_metrics(
            exposure_metrics, all_exposure_logs,
            os.path.join(save_dir, f"metrics_exposure_{save_name}_{controller_suffix}_{detection_guard}_{detection_guard_model_name}_{instruction_guard_name}.json")
        )
        logger.info(f"✓ Exposure metrics saved to {save_dir}")
    
    logger.info(f"\n✅ Exposure phase complete!")
    logger.info(f"   Processed {len(all_exposure_summary)} rounds")
    logger.info(f"   Remaining budget: {remain_budget}")
    
    return exposure_metrics, all_exposure_logs, remain_budget


# ========================================
# Main Function: TRIGGER Experiment
# ========================================
def main_rag_agent_trigger_experiment(
    model_name: str = "google/gemini-2.5-flash",
    dataset_name_or_path: str = "data-for-agents/insta-150k-v1",
    trigger_rounds: int = 10,
    attack_type: str = "completion_real",
    method_name: str = "zombie",
    budget_per_round: int = 10,
    total_budget: int = 20,
    max_steps: int = 10,
    detection_guard: bool = False,
    detection_guard_model_name: str = None,
    instruction_guard_name: str = "raw",
    attacker_model_name: str = "openai/gpt-4o-mini",
    controller_type: str = "pair",
    db_path: str = "/data2/xianglin/zombie_agent/db_storage/msmarco",
    top_k: int = 20,
    base_name: str = "base",
    exposure_name: str = "exposure",
    trigger_name: str = "trigger",
    evolve_mode: str = "raw",
    exposure_rounds: int = 10,
    save_dir: str = None,
    payload_dir: str = None,
    autodan_population_size: int = 20,
    autodan_num_elites: int = 2,
    autodan_mutation_rate: float = 0.1,
    autodan_crossover_rate: float = 0.5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Trigger-only experiment with adaptive attack optimization.
    Assumes exposure collection already exists in DB.
    
    Returns:
        Tuple of (trigger_metrics, all_trigger_logs)
    """
    # Payload directory (should be provided if running trigger-only after exposure)
    if payload_dir is None:
        logger.warning("No payload_dir provided for trigger-only experiment. Using default directory.")
        payload_dir = "src/tools/payloads"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"RAG TRIGGER EXPERIMENT (ADAPTIVE)")
    logger.info(f"{'='*80}")
    logger.info(f"Target Model: {model_name}")
    logger.info(f"Attacker Model: {attacker_model_name}")
    logger.info(f"Controller Type: {controller_type}")
    logger.info(f"Trigger Rounds: {trigger_rounds}")
    logger.info(f"Budget per Round: {budget_per_round}")
    logger.info(f"Total Budget: {total_budget}")
    logger.info(f"DB Path: {db_path}")
    logger.info(f"Top K: {top_k}")
    logger.info(f"Evolve Mode: {evolve_mode}")
    logger.info(f"Exposure Rounds (snapshot): {exposure_rounds}")
    logger.info(f"Payload Directory: {payload_dir}")
    logger.info(f"Detection guard: {detection_guard}")
    if detection_guard:
        logger.info(f"Detection guard model: {detection_guard_model_name}")
    logger.info(f"Instruction guard: {instruction_guard_name}")
    logger.info(f"{'='*80}\n")
    
    # Prepare queries
    trigger_queries = data_reader(dataset_name_or_path, num_questions=exposure_rounds+trigger_rounds)
    trigger_queries = trigger_queries[exposure_rounds:]
    trigger_queries = construct_trigger_queries(trigger_queries)
    
    # Initialize RAG memory and agent
    memory = RAGMemory(
        db_path=db_path,
        embedding_model="all-MiniLM-L6-v2",
        base_name=base_name,
        exposure_name=exposure_name,
        trigger_name=trigger_name,
        llm_model_name=model_name,
    )
    
    # Reset trigger collection only
    logger.info("\n[Trigger Mode] Resetting trigger collection...")
    memory.reset(targets="trigger")
    logger.info("✓ Trigger reset complete.\n")
    
    llm = load_model(model_name)
    agent = RAGWebAgent(
        llm=llm,
        memory=memory,
        max_steps=max_steps,
        detection_guard=detection_guard,
        detection_guard_model_name=detection_guard_model_name,
        instruction_guard_name=instruction_guard_name,
    )

    # Initialize Controller
    controller = create_controller(
        controller_type, agent, attacker_model_name,
        autodan_population_size, autodan_num_elites,
        autodan_mutation_rate, autodan_crossover_rate
    )
    
    # Run Trigger Phase
    trigger_metrics, all_trigger_logs, all_trigger_summary = trigger_RAG_adaptive(
        controller=controller,
        trigger_queries=trigger_queries,
        budget_per_round=budget_per_round,
        remain_budget=total_budget,
        exposure_rounds=exposure_rounds,
        evolve_mode=evolve_mode,
        top_k=top_k,
        payload_dir=payload_dir,
        detection_guard=detection_guard,
        detection_guard_model_name=detection_guard_model_name,
        instruction_guard_name=instruction_guard_name,
    )
    
    # Save results if save_dir provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_name = model_name.replace("/", "_") + f"_{method_name}" + f"_{attack_type}" + "_" + dataset_name_or_path.replace("/", "_")
        controller_suffix = controller_type.lower()
        
        save_trigger_metrics(
            trigger_metrics, all_trigger_logs,
            os.path.join(save_dir, f"metrics_trigger_{save_name}_{controller_suffix}_{detection_guard}_{detection_guard_model_name}_{instruction_guard_name}.json")
        )
        logger.info(f"✓ Trigger metrics saved to {save_dir}")
    
    logger.info(f"\n✅ Trigger phase complete!")
    logger.info(f"   Processed {len(all_trigger_summary)} rounds")
    
    return trigger_metrics, all_trigger_logs


# ========================================
# Main Function: BOTH Phases
# ========================================
def main_rag_agent_both_experiment(
    model_name: str = "google/gemini-2.5-flash",
    dataset_name_or_path: str = "data-for-agents/insta-150k-v1",
    exposure_rounds: int = 10,
    trigger_rounds: int = 10,
    attack_type: str = "completion_real",
    method_name: str = "zombie",
    budget_per_round: int = 10,
    total_budget: int = 20,
    max_steps: int = 10,
    detection_guard: bool = False,
    detection_guard_model_name: str = None,
    instruction_guard_name: str = "raw",
    attacker_model_name: str = "openai/gpt-4o-mini",
    controller_type: str = "pair",
    db_path: str = "/data2/xianglin/zombie_agent/db_storage/msmarco",
    top_k: int = 20,
    base_name: str = "base",
    exposure_name: str = "exposure",
    trigger_name: str = "trigger",
    evolve_mode: str = "raw",
    reset: bool = False,
    save_dir: str = "results/search_based_rag",
    payload_dir: str = None,
    autodan_population_size: int = 20,
    autodan_num_elites: int = 2,
    autodan_mutation_rate: float = 0.1,
    autodan_crossover_rate: float = 0.5,
) -> Dict[str, Any]:
    """
    Run both exposure and trigger phases sequentially.
    
    Returns:
        Dictionary with exposure and trigger summaries
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"RAG BOTH PHASES EXPERIMENT (ADAPTIVE)")
    logger.info(f"{'='*80}\n")
    
    # Phase 1: Exposure
    logger.info("\n" + "="*80)
    logger.info("RUNNING EXPOSURE PHASE")
    logger.info("="*80 + "\n")
    
    exposure_metrics, all_exposure_logs, remain_budget = main_rag_agent_exposure_experiment(
        model_name=model_name,
        dataset_name_or_path=dataset_name_or_path,
        exposure_rounds=exposure_rounds,
        attack_type=attack_type,
        method_name=method_name,
        budget_per_round=budget_per_round,
        total_budget=total_budget,
        max_steps=max_steps,
        detection_guard=detection_guard,
        detection_guard_model_name=detection_guard_model_name,
        instruction_guard_name=instruction_guard_name,
        attacker_model_name=attacker_model_name,
        controller_type=controller_type,
        db_path=db_path,
        top_k=top_k,
        base_name=base_name,
        exposure_name=exposure_name,
        trigger_name=trigger_name,
        evolve_mode=evolve_mode,
        reset=reset,
        save_dir=None,  # Don't save yet
        payload_dir=payload_dir,
        autodan_population_size=autodan_population_size,
        autodan_num_elites=autodan_num_elites,
        autodan_mutation_rate=autodan_mutation_rate,
        autodan_crossover_rate=autodan_crossover_rate,
    )
    
    # Phase 2: Trigger
    logger.info("\n" + "="*80)
    logger.info("RUNNING TRIGGER PHASE")
    logger.info("="*80 + "\n")
    
    trigger_metrics, all_trigger_logs = main_rag_agent_trigger_experiment(
        model_name=model_name,
        dataset_name_or_path=dataset_name_or_path,
        trigger_rounds=trigger_rounds,
        budget_per_round=budget_per_round,
        total_budget=remain_budget,  # Use remaining budget
        max_steps=max_steps,
        detection_guard=detection_guard,
        detection_guard_model_name=detection_guard_model_name,
        instruction_guard_name=instruction_guard_name,
        attacker_model_name=attacker_model_name,
        controller_type=controller_type,
        db_path=db_path,
        top_k=top_k,
        base_name=base_name,
        exposure_name=exposure_name,
        trigger_name=trigger_name,
        evolve_mode=evolve_mode,
        exposure_rounds=exposure_rounds,
        save_dir=None,  # Don't save yet
        payload_dir=payload_dir,
        autodan_population_size=autodan_population_size,
        autodan_num_elites=autodan_num_elites,
        autodan_mutation_rate=autodan_mutation_rate,
        autodan_crossover_rate=autodan_crossover_rate,
    )
    
    # Save combined results
    os.makedirs(save_dir, exist_ok=True)
    save_name = model_name.replace("/", "_") + f"_{method_name}" + f"_{attack_type}" + "_" + dataset_name_or_path.replace("/", "_")
    controller_suffix = controller_type.lower()
    
    save_exposure_metrics(
        exposure_metrics, all_exposure_logs,
        os.path.join(save_dir, f"metrics_exposure_{save_name}_{controller_suffix}_{detection_guard}_{detection_guard_model_name}_{instruction_guard_name}.json")
    )
    save_trigger_metrics(
        trigger_metrics, all_trigger_logs,
        os.path.join(save_dir, f"metrics_trigger_{save_name}_{controller_suffix}_{detection_guard}_{detection_guard_model_name}_{instruction_guard_name}.json")
    )
    plot_rag_metrics_multi_runs(
        [exposure_metrics], [trigger_metrics],
        save_path=os.path.join(save_dir, f"attack_{save_name}_{controller_suffix}_{detection_guard}_{detection_guard_model_name}_{instruction_guard_name}.png")
    )
    
    logger.info(f"\n✅ Both phases complete!")
    logger.info(f"   Results saved to {save_dir}")
    
    return {
        "exposure_metrics": exposure_metrics,
        "trigger_metrics": trigger_metrics,
    }


# ========================================
# Main Entry Point
# ========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive RAG attack with phase selection")
    
    # Phase control
    parser.add_argument("--phase", type=str, default="both",
                       choices=["exposure", "trigger", "both"],
                       help="Which phase to run: 'exposure', 'trigger', or 'both' (default)")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--attacker_model_name", type=str, default="openai/gpt-4o-mini")
    
    # Controller settings
    parser.add_argument("--controller_type", type=str, default="pair",
                       choices=["pair", "map_elites", "tap", "autodan"],
                       help="Type of search controller")
    
    # Dataset and rounds
    parser.add_argument("--dataset_name_or_path", type=str, default="data-for-agents/insta-150k-v1")
    parser.add_argument("--exposure_rounds", type=int, default=3)
    parser.add_argument("--trigger_rounds", type=int, default=50)
    parser.add_argument("--budget_per_round", type=int, default=10)
    parser.add_argument("--total_budget", type=int, default=20)
    
    # Attack settings
    parser.add_argument("--attack_type", type=str, default="completion_real", 
                       choices=["completion_real", "completion_realcmb", "completion_base64", 
                               "completion_2hash", "completion_1hash", "completion_0hash", 
                               "completion_upper", "completion_title", "completion_nospace", 
                               "completion_nocolon", "completion_typo", "completion_similar", 
                               "completion_ownlower", "completion_owntitle", "completion_ownhash", 
                               "completion_owndouble"])
    parser.add_argument("--method_name", type=str, default="zombie", 
                       choices=["dpi", "ipi", "zombie"])
    
    # Agent settings
    parser.add_argument("--max_steps", type=int, default=10)
    
    # RAG settings
    parser.add_argument("--db_path", type=str, default="/data2/xianglin/zombie_agent/db_storage/msmarco")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--base_name", type=str, default="base")
    parser.add_argument("--exposure_name", type=str, default="exposure")
    parser.add_argument("--trigger_name", type=str, default="trigger")
    parser.add_argument("--evolve_mode", type=str, default="raw")
    parser.add_argument("--reset", type=int, default=1)
    
    # Guard settings
    parser.add_argument("--detection_guard", type=int, default=0)
    parser.add_argument("--detection_guard_model_name", type=str, default="openai/gpt-4.1-nano")
    parser.add_argument("--instruction_guard_name", type=str, default="raw")
    
    # AutoDAN specific settings
    parser.add_argument("--autodan_population_size", type=int, default=20)
    parser.add_argument("--autodan_num_elites", type=int, default=2)
    parser.add_argument("--autodan_mutation_rate", type=float, default=0.1)
    parser.add_argument("--autodan_crossover_rate", type=float, default=0.5)
    
    # Save settings
    parser.add_argument("--save_dir", type=str, default="results/search_based_rag")
    parser.add_argument("--payload_dir", type=str, default=None,
                       help="Custom payload directory. If not provided, a unique directory will be created automatically.")
    
    args = parser.parse_args()

    setup_logging(task_name=f"{args.controller_type}_attack_rag_{args.phase}_{args.model_name.replace('/', '_')}")

    set_payload_dir(args.payload_dir)
    prepare_malicious_payload(args.method_name, args.attack_type)

    logger.info(f"\n{'='*80}")
    logger.info(f"EXPERIMENT CONFIGURATION")
    logger.info(f"{'='*80}")
    logger.info(f"Phase: {args.phase.upper()}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Controller: {args.controller_type}")
    logger.info(f"DB Path: {args.db_path}")
    if args.phase in ["exposure", "both"]:
        logger.info(f"Exposure rounds: {args.exposure_rounds}")
        logger.info(f"Reset mode: {'Yes' if args.reset else 'Resume'}")
    if args.phase in ["trigger", "both"]:
        logger.info(f"Trigger rounds: {args.trigger_rounds}")
        logger.info(f"Dataset: {args.dataset_name_or_path}")
    logger.info(f"{'='*80}\n")

    # Run based on phase selection
    if args.phase == "exposure":
        main_rag_agent_exposure_experiment(
            model_name=args.model_name,
            dataset_name_or_path=args.dataset_name_or_path,
            exposure_rounds=args.exposure_rounds,
            attack_type=args.attack_type,
            method_name=args.method_name,
            budget_per_round=args.budget_per_round,
            total_budget=args.total_budget,
            max_steps=args.max_steps,
            detection_guard=bool(args.detection_guard),
            detection_guard_model_name=args.detection_guard_model_name,
            instruction_guard_name=args.instruction_guard_name,
            attacker_model_name=args.attacker_model_name,
            controller_type=args.controller_type,
            db_path=args.db_path,
            top_k=args.top_k,
            base_name=args.base_name,
            exposure_name=args.exposure_name,
            trigger_name=args.trigger_name,
            evolve_mode=args.evolve_mode,
            reset=bool(args.reset),
            save_dir=args.save_dir,
            autodan_population_size=args.autodan_population_size,
            autodan_num_elites=args.autodan_num_elites,
            autodan_mutation_rate=args.autodan_mutation_rate,
            autodan_crossover_rate=args.autodan_crossover_rate,
        )
    elif args.phase == "trigger":
        main_rag_agent_trigger_experiment(
            model_name=args.model_name,
            dataset_name_or_path=args.dataset_name_or_path,
            trigger_rounds=args.trigger_rounds,
            attack_type=args.attack_type,
            method_name=args.method_name,
            budget_per_round=args.budget_per_round,
            total_budget=args.total_budget,
            max_steps=args.max_steps,
            detection_guard=bool(args.detection_guard),
            detection_guard_model_name=args.detection_guard_model_name,
            instruction_guard_name=args.instruction_guard_name,
            attacker_model_name=args.attacker_model_name,
            controller_type=args.controller_type,
            db_path=args.db_path,
            top_k=args.top_k,
            base_name=args.base_name,
            exposure_name=args.exposure_name,
            trigger_name=args.trigger_name,
            evolve_mode=args.evolve_mode,
            exposure_rounds=args.exposure_rounds,
            save_dir=args.save_dir,
            autodan_population_size=args.autodan_population_size,
            autodan_num_elites=args.autodan_num_elites,
            autodan_mutation_rate=args.autodan_mutation_rate,
            autodan_crossover_rate=args.autodan_crossover_rate,
        )
    else:  # both
        main_rag_agent_both_experiment(
            model_name=args.model_name,
            dataset_name_or_path=args.dataset_name_or_path,
            exposure_rounds=args.exposure_rounds,
            trigger_rounds=args.trigger_rounds,
            attack_type=args.attack_type,
            method_name=args.method_name,
            budget_per_round=args.budget_per_round,
            total_budget=args.total_budget,
            max_steps=args.max_steps,
            detection_guard=bool(args.detection_guard),
            detection_guard_model_name=args.detection_guard_model_name,
            instruction_guard_name=args.instruction_guard_name,
            attacker_model_name=args.attacker_model_name,
            controller_type=args.controller_type,
            db_path=args.db_path,
            top_k=args.top_k,
            base_name=args.base_name,
            exposure_name=args.exposure_name,
            trigger_name=args.trigger_name,
            evolve_mode=args.evolve_mode,
            reset=bool(args.reset),
            save_dir=args.save_dir,
            autodan_population_size=args.autodan_population_size,
            autodan_num_elites=args.autodan_num_elites,
            autodan_mutation_rate=args.autodan_mutation_rate,
            autodan_crossover_rate=args.autodan_crossover_rate,
        )

    logger.info("\n✅ Done.")

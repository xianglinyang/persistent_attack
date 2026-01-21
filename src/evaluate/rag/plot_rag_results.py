#!/usr/bin/env python3
"""
Example script for plotting RAG attack results.

Usage:
    # Single run - automatic when running with --phase both
    python src/evaluate/rag/run_attack.py --phase both --model_name google/gemini-2.5-flash
    
    # Or manually plot existing results
    python scripts/plot_rag_results.py
"""

import sys
sys.path.append('.')

from src.evaluate.plot_results import plot_rag_metrics
from src.evaluate.plot_rag_multi_runs import plot_rag_metrics_multi_runs

# ============================================================
# Example 1: Plot single run results
# ============================================================
def plot_single_run(exposure_metrics, trigger_metrics, model_name="model"):
    """Plot results from a single run."""
    
    # Flatten trigger_metrics if it's a list of batches
    if trigger_metrics and isinstance(trigger_metrics[0], list):
        trigger_metrics_flat = []
        for batch in trigger_metrics:
            trigger_metrics_flat.extend(batch)
    else:
        trigger_metrics_flat = trigger_metrics
    
    results = {
        "exposure_metrics": exposure_metrics,
        "trigger_metrics": trigger_metrics_flat
    }
    
    save_path = f"results/{model_name}_rag_attack_metrics.png"
    plot_rag_metrics(results, save_path=save_path)
    print(f"✅ Single run plot saved to: {save_path}")


# ============================================================
# Example 2: Plot multiple runs with mean and std
# ============================================================
def plot_multiple_runs(exposure_metrics_list, trigger_metrics_list, model_name="model", show_individual=False):
    """Plot results from multiple runs with mean and std."""
    
    save_path = f"results/{model_name}_rag_attack_metrics_multi_runs.png"
    plot_rag_metrics_multi_runs(
        exposure_metrics_list, 
        trigger_metrics_list, 
        save_path=save_path,
        show_individual=show_individual
    )
    print(f"✅ Multi-run plot saved to: {save_path}")


# ============================================================
# Example usage with mock data
# ============================================================
if __name__ == "__main__":
    import numpy as np
    
    print("\n" + "="*80)
    print("RAG ATTACK PLOTTING EXAMPLES")
    print("="*80 + "\n")
    
    # Create mock data for demonstration
    n_exposure = 10
    n_trigger = 5
    n_runs = 3
    
    exposure_metrics_list = []
    trigger_metrics_list = []
    
    for run in range(n_runs):
        # Mock exposure metrics
        exposure_metrics = []
        for i in range(n_exposure):
            exposure_metrics.append({
                "exposure_round": i + 1,
                "rag_payload_count": np.random.randint(1, 5),
                "recall@10": np.random.random(),
                "recall@50": np.random.random(),
                "recall@100": np.random.random(),
            })
        exposure_metrics_list.append(exposure_metrics)
        
        # Mock trigger metrics (as batches)
        trigger_batch = []
        for i in range(n_trigger):
            trigger_batch.append({
                "trigger_round": i + 1,
                "exposure_round": n_exposure,
                "rag_payload_count": np.random.randint(1, 5),
                "recall@10": np.random.random(),
                "recall@50": np.random.random(),
                "recall@100": np.random.random(),
                "exfiltration": np.random.choice([True, False]),
                "command_exec": np.random.choice([True, False]),
                "reload_payload_times": np.random.randint(0, 3),
            })
        trigger_metrics_list.append([trigger_batch])  # As list of batches
    
    # Example 1: Plot single run
    print("\n[Example 1] Plotting single run...")
    plot_single_run(
        exposure_metrics_list[0], 
        trigger_metrics_list[0], 
        model_name="example_single"
    )
    
    # Example 2: Plot multiple runs
    print("\n[Example 2] Plotting multiple runs with mean and std...")
    plot_multiple_runs(
        exposure_metrics_list, 
        trigger_metrics_list, 
        model_name="example_multi",
        show_individual=True  # Show individual runs as faint lines
    )
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
    print("\n📊 Check the results/ directory for generated plots")
    print("\n💡 Usage with real data:")
    print("   1. Run attack: python src/evaluate/rag/run_attack.py --phase both")
    print("   2. Plots are automatically generated when --phase both is used")
    print("   3. For multiple runs, collect exposure_metrics and trigger_metrics")
    print("      from each run and use plot_rag_metrics_multi_runs()")
    print("="*80 + "\n")

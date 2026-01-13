import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def plot_sliding_window_metrics(exposure_metrics: List[Dict], trigger_metrics: List[Dict], save_path: Optional[str] = None):
    """
    Plot memory persistence rate and ASR (separated by exfiltration and command_exec) over rounds for sliding window attack.
    
    Args:
        results: Results dictionary from run_sliding_window_attack()
        save_path: Optional path to save the plot (e.g., 'attack_metrics.png')
    """
    # Extract data
    exposure_rounds = [m["exposure_round"] for m in exposure_metrics]
    exposure_persistence = [1 if m["payload_in_memory"] else 0 for m in exposure_metrics]
    
    trigger_rounds = [m["trigger_round"] + len(exposure_rounds) for m in trigger_metrics]
    trigger_persistence = [1 if m["payload_in_memory"] else 0 for m in trigger_metrics]
    trigger_exfiltration = [1 if m.get("exfiltration", False) else 0 for m in trigger_metrics]
    trigger_command_exec = [1 if m.get("command_exec", False) else 0 for m in trigger_metrics]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Memory Persistence Rate
    ax1.plot(exposure_rounds, exposure_persistence, 'o-', 
             label='Exposure Phase', color='blue', linewidth=2, markersize=8)
    ax1.plot(trigger_rounds, trigger_persistence, 's-', 
             label='Trigger Phase', color='orange', linewidth=2, markersize=8)
    ax1.axvline(x=len(exposure_rounds), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Payload in Memory (0/1)', fontsize=12)
    ax1.set_title('Memory Persistence Rate Over Rounds', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot 2: Exfiltration ASR (only trigger phase)
    ax2.plot(trigger_rounds, trigger_exfiltration, 's-', 
             label='Exfiltration ASR', color='red', linewidth=2, markersize=8)
    ax2.axvline(x=len(exposure_rounds), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Exfiltration Success (0/1)', fontsize=12)
    ax2.set_title('Exfiltration ASR Over Rounds', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # Plot 3: Command Execution ASR (only trigger phase)
    ax3.plot(trigger_rounds, trigger_command_exec, 's-', 
             label='Command Exec ASR', color='purple', linewidth=2, markersize=8)
    ax3.axvline(x=len(exposure_rounds), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Command Exec Success (0/1)', fontsize=12)
    ax3.set_title('Command Execution ASR Over Rounds', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[Plot saved to {save_path}]")
    
    plt.show()


def plot_sliding_window_metrics_multi_runs(
    exposure_metrics_list: List[Dict], trigger_metrics_list: List[Dict], 
    save_path: Optional[str] = None,
    show_individual: bool = False
):
    """
    Plot memory persistence rate and ASR with mean and std across multiple runs.
    
    Args:
        results_list: List of results dictionaries from multiple run_sliding_window_attack() calls
        save_path: Optional path to save the plot
        show_individual: Whether to show individual run lines (faint) in addition to mean
    """
    
    # Determine the number of rounds from the first result
    n_exposure = len(exposure_metrics_list[0])
    n_trigger = len(trigger_metrics_list[0])
    n_runs = len(exposure_metrics_list)
    
    # Initialize arrays to store metrics across runs
    # Shape: (n_runs, n_rounds)
    exposure_persistence_all = np.zeros((n_runs, n_exposure))
    trigger_persistence_all = np.zeros((n_runs, n_trigger))
    trigger_exfiltration_all = np.zeros((n_runs, n_trigger))
    trigger_command_exec_all = np.zeros((n_runs, n_trigger))
    
    # Collect data from all runs
    for run_idx, (exposure_metrics, trigger_metrics) in enumerate(zip(exposure_metrics_list, trigger_metrics_list)):
        
        for i, m in enumerate(exposure_metrics):
            exposure_persistence_all[run_idx, i] = 1 if m["payload_in_memory"] else 0
        
        for i, m in enumerate(trigger_metrics):
            trigger_persistence_all[run_idx, i] = 1 if m["payload_in_memory"] else 0
            trigger_exfiltration_all[run_idx, i] = 1 if m.get("exfiltration", False) else 0
            trigger_command_exec_all[run_idx, i] = 1 if m.get("command_exec", False) else 0
    
    # Compute mean and std
    exposure_persistence_mean = np.mean(exposure_persistence_all, axis=0)
    exposure_persistence_std = np.std(exposure_persistence_all, axis=0)
    
    trigger_persistence_mean = np.mean(trigger_persistence_all, axis=0)
    trigger_persistence_std = np.std(trigger_persistence_all, axis=0)
    
    trigger_exfiltration_mean = np.mean(trigger_exfiltration_all, axis=0)
    trigger_exfiltration_std = np.std(trigger_exfiltration_all, axis=0)
    
    trigger_command_exec_mean = np.mean(trigger_command_exec_all, axis=0)
    trigger_command_exec_std = np.std(trigger_command_exec_all, axis=0)
    
    # Create round indices
    exposure_rounds = np.arange(1, n_exposure + 1)
    trigger_rounds = np.arange(n_exposure + 1, n_exposure + n_trigger + 1)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Memory Persistence Rate
    if show_individual:
        for run_idx in range(n_runs):
            ax1.plot(exposure_rounds, exposure_persistence_all[run_idx], 'o-', 
                    color='blue', alpha=0.2, linewidth=1, markersize=4)
            ax1.plot(trigger_rounds, trigger_persistence_all[run_idx], 's-', 
                    color='orange', alpha=0.2, linewidth=1, markersize=4)
    
    ax1.plot(exposure_rounds, exposure_persistence_mean, 'o-', 
             label=f'Exposure Phase (n={n_runs})', color='blue', linewidth=2, markersize=8)
    ax1.fill_between(exposure_rounds, 
                     exposure_persistence_mean - exposure_persistence_std,
                     exposure_persistence_mean + exposure_persistence_std,
                     color='blue', alpha=0.2)
    
    ax1.plot(trigger_rounds, trigger_persistence_mean, 's-', 
             label=f'Trigger Phase (n={n_runs})', color='orange', linewidth=2, markersize=8)
    ax1.fill_between(trigger_rounds, 
                     trigger_persistence_mean - trigger_persistence_std,
                     trigger_persistence_mean + trigger_persistence_std,
                     color='orange', alpha=0.2)
    
    ax1.axvline(x=n_exposure, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Payload in Memory', fontsize=12)
    ax1.set_title('Memory Persistence Rate Over Rounds\n(Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot 2: Exfiltration ASR
    if show_individual:
        for run_idx in range(n_runs):
            ax2.plot(trigger_rounds, trigger_exfiltration_all[run_idx], 's-', 
                    color='red', alpha=0.2, linewidth=1, markersize=4)
    
    ax2.plot(trigger_rounds, trigger_exfiltration_mean, 's-', 
             label=f'Exfiltration ASR (n={n_runs})', color='red', linewidth=2, markersize=8)
    ax2.fill_between(trigger_rounds, 
                     trigger_exfiltration_mean - trigger_exfiltration_std,
                     trigger_exfiltration_mean + trigger_exfiltration_std,
                     color='red', alpha=0.2)
    
    ax2.axvline(x=n_exposure, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Exfiltration Success', fontsize=12)
    ax2.set_title('Exfiltration ASR Over Rounds\n(Mean ± Std)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # Plot 3: Command Execution ASR
    if show_individual:
        for run_idx in range(n_runs):
            ax3.plot(trigger_rounds, trigger_command_exec_all[run_idx], 's-', 
                    color='purple', alpha=0.2, linewidth=1, markersize=4)
    
    ax3.plot(trigger_rounds, trigger_command_exec_mean, 's-', 
             label=f'Command Exec ASR (n={n_runs})', color='purple', linewidth=2, markersize=8)
    ax3.fill_between(trigger_rounds, 
                     trigger_command_exec_mean - trigger_command_exec_std,
                     trigger_command_exec_mean + trigger_command_exec_std,
                     color='purple', alpha=0.2)
    
    ax3.axvline(x=n_exposure, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Command Exec Success', fontsize=12)
    ax3.set_title('Command Execution ASR Over Rounds\n(Mean ± Std)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[Plot saved to {save_path}]")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Summary Statistics (n={n_runs} runs)")
    print(f"{'='*80}")
    print(f"\nTrigger Phase ASR:")
    print(f"  Exfiltration: {trigger_exfiltration_mean.mean():.2%} ± {trigger_exfiltration_std.mean():.2%}")
    print(f"  Command Exec: {trigger_command_exec_mean.mean():.2%} ± {trigger_command_exec_std.mean():.2%}")
    print(f"\nMemory Persistence:")
    print(f"  Exposure: {exposure_persistence_mean.mean():.2%} ± {exposure_persistence_std.mean():.2%}")
    print(f"  Trigger: {trigger_persistence_mean.mean():.2%} ± {trigger_persistence_std.mean():.2%}")
    print(f"{'='*80}\n")


def plot_rag_metrics(results: Dict, save_path: Optional[str] = None):
    """
    Plot memory persistence (payload count), recall@k, and ASR (separated by exfiltration and command_exec) over rounds for RAG attack.
    
    Args:
        results: Results dictionary from run_rag_attack()
        save_path: Optional path to save the plot (e.g., 'rag_attack_metrics.png')
    """
    # Extract data
    exposure_metrics = results["exposure_metrics"]
    trigger_metrics = results["trigger_metrics"]
    
    exposure_rounds = [m["exposure_round"] for m in exposure_metrics]
    exposure_payload_count = [m["rag_payload_count"] for m in exposure_metrics]
    exposure_recall_10 = [1 if m["recall@10"] else 0 for m in exposure_metrics]
    
    trigger_rounds = [m["trigger_round"] + len(exposure_rounds) for m in trigger_metrics]
    trigger_payload_count = [m["rag_payload_count"] for m in trigger_metrics]
    trigger_recall_10 = [1 if m["recall@10"] else 0 for m in trigger_metrics]
    trigger_exfiltration = [1 if m.get("exfiltration", False) else 0 for m in trigger_metrics]
    trigger_command_exec = [1 if m.get("command_exec", False) else 0 for m in trigger_metrics]
    
    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Payload Count
    ax1.plot(exposure_rounds, exposure_payload_count, 'o-', 
             label='Exposure Phase', color='blue', linewidth=2, markersize=8)
    ax1.plot(trigger_rounds, trigger_payload_count, 's-', 
             label='Trigger Phase', color='orange', linewidth=2, markersize=8)
    ax1.axvline(x=len(exposure_rounds), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Payload Count in RAG Memory', fontsize=12)
    ax1.set_title('RAG Payload Count Over Rounds', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recall@10
    ax2.plot(exposure_rounds, exposure_recall_10, 'o-', 
             label='Exposure Phase', color='blue', linewidth=2, markersize=8)
    ax2.plot(trigger_rounds, trigger_recall_10, 's-', 
             label='Trigger Phase', color='orange', linewidth=2, markersize=8)
    ax2.axvline(x=len(exposure_rounds), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Recall@10 (0/1)', fontsize=12)
    ax2.set_title('Recall@10 Over Rounds', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # Plot 3: Exfiltration ASR (only trigger phase)
    ax3.plot(trigger_rounds, trigger_exfiltration, 's-', 
             label='Exfiltration ASR', color='red', linewidth=2, markersize=8)
    ax3.axvline(x=len(exposure_rounds), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Exfiltration Success (0/1)', fontsize=12)
    ax3.set_title('Exfiltration ASR Over Rounds', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    
    # Plot 4: Command Execution ASR (only trigger phase)
    ax4.plot(trigger_rounds, trigger_command_exec, 's-', 
             label='Command Exec ASR', color='purple', linewidth=2, markersize=8)
    ax4.axvline(x=len(exposure_rounds), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Command Exec Success (0/1)', fontsize=12)
    ax4.set_title('Command Execution ASR Over Rounds', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[Plot saved to {save_path}]")
    
    plt.show()

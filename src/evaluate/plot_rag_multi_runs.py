"""
Plot RAG attack metrics for multiple runs with mean and std.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def plot_rag_metrics_multi_runs(
    exposure_metrics_list: List[List[Dict]], 
    trigger_metrics_list: List[List[Dict]], 
    save_path: Optional[str] = None,
    show_individual: bool = False
):
    """
    Plot RAG metrics with mean and std across multiple runs.
    
    Args:
        exposure_metrics_list: List of exposure_metrics from multiple runs
        trigger_metrics_list: List of trigger_metrics from multiple runs (each is a list of batches or flat list)
        save_path: Optional path to save the plot
        show_individual: Whether to show individual run lines (faint) in addition to mean
    """
    
    # Flatten trigger metrics if they contain batches
    trigger_metrics_list_flat = []
    for trigger_metrics in trigger_metrics_list:
        if trigger_metrics and isinstance(trigger_metrics[0], list):
            # It's a list of batches, flatten it
            flat = []
            for batch in trigger_metrics:
                flat.extend(batch)
            trigger_metrics_list_flat.append(flat)
        else:
            # Already flat
            trigger_metrics_list_flat.append(trigger_metrics)
    
    # Determine the number of rounds from the first result
    n_exposure = len(exposure_metrics_list[0])
    n_trigger = len(trigger_metrics_list_flat[0]) if trigger_metrics_list_flat[0] else 0
    n_runs = len(exposure_metrics_list)
    
    # Initialize arrays to store metrics across runs
    exposure_payload_all = np.zeros((n_runs, n_exposure))
    exposure_recall_10_all = np.zeros((n_runs, n_exposure))
    exposure_recall_50_all = np.zeros((n_runs, n_exposure))
    exposure_recall_100_all = np.zeros((n_runs, n_exposure))
    
    trigger_payload_all = np.zeros((n_runs, n_trigger)) if n_trigger > 0 else None
    trigger_recall_10_all = np.zeros((n_runs, n_trigger)) if n_trigger > 0 else None
    trigger_recall_50_all = np.zeros((n_runs, n_trigger)) if n_trigger > 0 else None
    trigger_recall_100_all = np.zeros((n_runs, n_trigger)) if n_trigger > 0 else None
    trigger_exfiltration_all = np.zeros((n_runs, n_trigger)) if n_trigger > 0 else None
    trigger_command_exec_all = np.zeros((n_runs, n_trigger)) if n_trigger > 0 else None
    trigger_reload_times_all = np.zeros((n_runs, n_trigger)) if n_trigger > 0 else None
    
    # Collect data from all runs
    for run_idx, exposure_metrics in enumerate(exposure_metrics_list):
        for i, m in enumerate(exposure_metrics):
            exposure_payload_all[run_idx, i] = m.get("rag_payload_count", 0)
            exposure_recall_10_all[run_idx, i] = m.get("recall@10", 0)
            exposure_recall_50_all[run_idx, i] = m.get("recall@50", 0)
            exposure_recall_100_all[run_idx, i] = m.get("recall@100", 0)
    
    if n_trigger > 0:
        for run_idx, trigger_metrics in enumerate(trigger_metrics_list_flat):
            for i, m in enumerate(trigger_metrics):
                trigger_payload_all[run_idx, i] = m.get("rag_payload_count", 0)
                trigger_recall_10_all[run_idx, i] = m.get("recall@10", 0)
                trigger_recall_50_all[run_idx, i] = m.get("recall@50", 0)
                trigger_recall_100_all[run_idx, i] = m.get("recall@100", 0)
                trigger_exfiltration_all[run_idx, i] = 1 if m.get("exfiltration", False) else 0
                trigger_command_exec_all[run_idx, i] = 1 if m.get("command_exec", False) else 0
                trigger_reload_times_all[run_idx, i] = m.get("reload_payload_times", 0)
    
    # Compute mean and std for exposure
    exposure_payload_mean = np.mean(exposure_payload_all, axis=0)
    exposure_payload_std = np.std(exposure_payload_all, axis=0)
    exposure_recall_10_mean = np.mean(exposure_recall_10_all, axis=0)
    exposure_recall_10_std = np.std(exposure_recall_10_all, axis=0)
    exposure_recall_50_mean = np.mean(exposure_recall_50_all, axis=0)
    exposure_recall_50_std = np.std(exposure_recall_50_all, axis=0)
    exposure_recall_100_mean = np.mean(exposure_recall_100_all, axis=0)
    exposure_recall_100_std = np.std(exposure_recall_100_all, axis=0)
    
    # Compute mean and std for trigger (if available)
    if n_trigger > 0:
        trigger_payload_mean = np.mean(trigger_payload_all, axis=0)
        trigger_payload_std = np.std(trigger_payload_all, axis=0)
        trigger_recall_10_mean = np.mean(trigger_recall_10_all, axis=0)
        trigger_recall_10_std = np.std(trigger_recall_10_all, axis=0)
        trigger_recall_50_mean = np.mean(trigger_recall_50_all, axis=0)
        trigger_recall_50_std = np.std(trigger_recall_50_all, axis=0)
        trigger_recall_100_mean = np.mean(trigger_recall_100_all, axis=0)
        trigger_recall_100_std = np.std(trigger_recall_100_all, axis=0)
        trigger_exfiltration_mean = np.mean(trigger_exfiltration_all, axis=0)
        trigger_exfiltration_std = np.std(trigger_exfiltration_all, axis=0)
        trigger_command_exec_mean = np.mean(trigger_command_exec_all, axis=0)
        trigger_command_exec_std = np.std(trigger_command_exec_all, axis=0)
        trigger_reload_times_mean = np.mean(trigger_reload_times_all, axis=0)
        trigger_reload_times_std = np.std(trigger_reload_times_all, axis=0)
    
    # Create round indices
    exposure_rounds = np.arange(1, n_exposure + 1)
    trigger_rounds = np.arange(n_exposure + 1, n_exposure + n_trigger + 1) if n_trigger > 0 else None
    
    # Create figure with 6 subplots (3x2)
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    
    # Plot 1: Payload Count
    if show_individual:
        for run_idx in range(n_runs):
            ax1.plot(exposure_rounds, exposure_payload_all[run_idx], 'o-', 
                    color='blue', alpha=0.2, linewidth=1, markersize=4)
            if n_trigger > 0:
                ax1.plot(trigger_rounds, trigger_payload_all[run_idx], 's-', 
                        color='orange', alpha=0.2, linewidth=1, markersize=4)
    
    ax1.plot(exposure_rounds, exposure_payload_mean, 'o-', 
             label=f'Exposure Phase (n={n_runs})', color='blue', linewidth=2, markersize=8)
    ax1.fill_between(exposure_rounds, 
                     exposure_payload_mean - exposure_payload_std,
                     exposure_payload_mean + exposure_payload_std,
                     color='blue', alpha=0.2)
    
    if n_trigger > 0:
        ax1.plot(trigger_rounds, trigger_payload_mean, 's-', 
                 label=f'Trigger Phase (n={n_runs})', color='orange', linewidth=2, markersize=8)
        ax1.fill_between(trigger_rounds, 
                         trigger_payload_mean - trigger_payload_std,
                         trigger_payload_mean + trigger_payload_std,
                         color='orange', alpha=0.2)
    
    ax1.axvline(x=n_exposure, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Payload Count', fontsize=12)
    ax1.set_title('RAG Payload Count Over Rounds\n(Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recall@10
    if show_individual:
        for run_idx in range(n_runs):
            ax2.plot(exposure_rounds, exposure_recall_10_all[run_idx], 'o-', 
                    color='blue', alpha=0.2, linewidth=1, markersize=4)
            if n_trigger > 0:
                ax2.plot(trigger_rounds, trigger_recall_10_all[run_idx], 's-', 
                        color='orange', alpha=0.2, linewidth=1, markersize=4)
    
    ax2.plot(exposure_rounds, exposure_recall_10_mean, 'o-', 
             label=f'Exposure Phase (n={n_runs})', color='blue', linewidth=2, markersize=8)
    ax2.fill_between(exposure_rounds, 
                     exposure_recall_10_mean - exposure_recall_10_std,
                     exposure_recall_10_mean + exposure_recall_10_std,
                     color='blue', alpha=0.2)
    
    if n_trigger > 0:
        ax2.plot(trigger_rounds, trigger_recall_10_mean, 's-', 
                 label=f'Trigger Phase (n={n_runs})', color='orange', linewidth=2, markersize=8)
        ax2.fill_between(trigger_rounds, 
                         trigger_recall_10_mean - trigger_recall_10_std,
                         trigger_recall_10_mean + trigger_recall_10_std,
                         color='orange', alpha=0.2)
    
    ax2.axvline(x=n_exposure, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Recall@10', fontsize=12)
    ax2.set_title('Recall@10 Over Rounds\n(Mean ± Std)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Recall@50
    if show_individual:
        for run_idx in range(n_runs):
            ax3.plot(exposure_rounds, exposure_recall_50_all[run_idx], 'o-', 
                    color='blue', alpha=0.2, linewidth=1, markersize=4)
            if n_trigger > 0:
                ax3.plot(trigger_rounds, trigger_recall_50_all[run_idx], 's-', 
                        color='orange', alpha=0.2, linewidth=1, markersize=4)
    
    ax3.plot(exposure_rounds, exposure_recall_50_mean, 'o-', 
             label=f'Exposure Phase (n={n_runs})', color='blue', linewidth=2, markersize=8)
    ax3.fill_between(exposure_rounds, 
                     exposure_recall_50_mean - exposure_recall_50_std,
                     exposure_recall_50_mean + exposure_recall_50_std,
                     color='blue', alpha=0.2)
    
    if n_trigger > 0:
        ax3.plot(trigger_rounds, trigger_recall_50_mean, 's-', 
                 label=f'Trigger Phase (n={n_runs})', color='orange', linewidth=2, markersize=8)
        ax3.fill_between(trigger_rounds, 
                         trigger_recall_50_mean - trigger_recall_50_std,
                         trigger_recall_50_mean + trigger_recall_50_std,
                         color='orange', alpha=0.2)
    
    ax3.axvline(x=n_exposure, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Recall@50', fontsize=12)
    ax3.set_title('Recall@50 Over Rounds\n(Mean ± Std)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Recall@100
    if show_individual:
        for run_idx in range(n_runs):
            ax4.plot(exposure_rounds, exposure_recall_100_all[run_idx], 'o-', 
                    color='blue', alpha=0.2, linewidth=1, markersize=4)
            if n_trigger > 0:
                ax4.plot(trigger_rounds, trigger_recall_100_all[run_idx], 's-', 
                        color='orange', alpha=0.2, linewidth=1, markersize=4)
    
    ax4.plot(exposure_rounds, exposure_recall_100_mean, 'o-', 
             label=f'Exposure Phase (n={n_runs})', color='blue', linewidth=2, markersize=8)
    ax4.fill_between(exposure_rounds, 
                     exposure_recall_100_mean - exposure_recall_100_std,
                     exposure_recall_100_mean + exposure_recall_100_std,
                     color='blue', alpha=0.2)
    
    if n_trigger > 0:
        ax4.plot(trigger_rounds, trigger_recall_100_mean, 's-', 
                 label=f'Trigger Phase (n={n_runs})', color='orange', linewidth=2, markersize=8)
        ax4.fill_between(trigger_rounds, 
                         trigger_recall_100_mean - trigger_recall_100_std,
                         trigger_recall_100_mean + trigger_reload_times_std,
                         color='orange', alpha=0.2)
    
    ax4.axvline(x=n_exposure, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Recall@100', fontsize=12)
    ax4.set_title('Recall@100 Over Rounds\n(Mean ± Std)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: ASR - Exfiltration & Command Exec (trigger phase only)
    if n_trigger > 0:
        if show_individual:
            for run_idx in range(n_runs):
                ax5.plot(trigger_rounds, trigger_exfiltration_all[run_idx], 's-', 
                        color='red', alpha=0.2, linewidth=1, markersize=4)
                ax5.plot(trigger_rounds, trigger_command_exec_all[run_idx], '^-', 
                        color='purple', alpha=0.2, linewidth=1, markersize=4)
        
        ax5.plot(trigger_rounds, trigger_exfiltration_mean, 's-', 
                 label=f'Exfiltration (n={n_runs})', color='red', linewidth=2, markersize=8)
        ax5.fill_between(trigger_rounds, 
                         trigger_exfiltration_mean - trigger_exfiltration_std,
                         trigger_exfiltration_mean + trigger_exfiltration_std,
                         color='red', alpha=0.2)
        
        ax5.plot(trigger_rounds, trigger_command_exec_mean, '^-', 
                 label=f'Command Exec (n={n_runs})', color='purple', linewidth=2, markersize=8)
        ax5.fill_between(trigger_rounds, 
                         trigger_command_exec_mean - trigger_command_exec_std,
                         trigger_command_exec_mean + trigger_command_exec_std,
                         color='purple', alpha=0.2)
        
        ax5.axvline(x=n_exposure, color='red', linestyle='--', 
                    linewidth=2, alpha=0.7, label='Phase Transition')
        ax5.set_xlabel('Round', fontsize=12)
        ax5.set_ylabel('Attack Success Rate', fontsize=12)
        ax5.set_title('ASR Over Rounds\n(Mean ± Std)', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(-0.1, 1.1)
    else:
        ax5.text(0.5, 0.5, 'No Trigger Data', ha='center', va='center', fontsize=16)
        ax5.axis('off')
    
    # Plot 6: Reload Payload Times (trigger phase only)
    if n_trigger > 0:
        if show_individual:
            for run_idx in range(n_runs):
                ax6.plot(trigger_rounds, trigger_reload_times_all[run_idx], 's-', 
                        color='green', alpha=0.2, linewidth=1, markersize=4)
        
        ax6.plot(trigger_rounds, trigger_reload_times_mean, 's-', 
                 label=f'Reload Times (n={n_runs})', color='green', linewidth=2, markersize=8)
        ax6.fill_between(trigger_rounds, 
                         trigger_reload_times_mean - trigger_reload_times_std,
                         trigger_reload_times_mean + trigger_reload_times_std,
                         color='green', alpha=0.2)
        
        ax6.axvline(x=n_exposure, color='red', linestyle='--', 
                    linewidth=2, alpha=0.7, label='Phase Transition')
        ax6.set_xlabel('Round', fontsize=12)
        ax6.set_ylabel('Reload Payload Times', fontsize=12)
        ax6.set_title('Payload Reload Times Over Rounds\n(Mean ± Std)', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No Trigger Data', ha='center', va='center', fontsize=16)
        ax6.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[Plot saved to {save_path}]")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Summary Statistics (n={n_runs} runs)")
    print(f"{'='*80}")
    
    print(f"\nExposure Phase:")
    print(f"  Avg Payload Count: {exposure_payload_mean.mean():.2f} ± {exposure_payload_std.mean():.2f}")
    print(f"  Avg Recall@10: {exposure_recall_10_mean.mean():.2f} ± {exposure_recall_10_std.mean():.2f}")
    print(f"  Avg Recall@50: {exposure_recall_50_mean.mean():.2f} ± {exposure_recall_50_std.mean():.2f}")
    print(f"  Avg Recall@100: {exposure_recall_100_mean.mean():.2f} ± {exposure_recall_100_std.mean():.2f}")
    
    if n_trigger > 0:
        print(f"\nTrigger Phase:")
        print(f"  Avg Payload Count: {trigger_payload_mean.mean():.2f} ± {trigger_payload_std.mean():.2f}")
        print(f"  Avg Recall@10: {trigger_recall_10_mean.mean():.2f} ± {trigger_recall_10_std.mean():.2f}")
        print(f"  Avg Recall@50: {trigger_recall_50_mean.mean():.2f} ± {trigger_recall_50_std.mean():.2f}")
        print(f"  Avg Recall@100: {trigger_recall_100_mean.mean():.2f} ± {trigger_recall_100_std.mean():.2f}")
        print(f"  Exfiltration ASR: {trigger_exfiltration_mean.mean():.2%} ± {trigger_exfiltration_std.mean():.2%}")
        print(f"  Command Exec ASR: {trigger_command_exec_mean.mean():.2%} ± {trigger_command_exec_std.mean():.2%}")
        print(f"  Reload Payload Times: {trigger_reload_times_mean.mean():.2f} ± {trigger_reload_times_std.mean():.2f}")
    
    print(f"{'='*80}\n")

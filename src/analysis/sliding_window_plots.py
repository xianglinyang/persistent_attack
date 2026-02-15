'''
File to plot the metrics from RAG agent attacks with sliding window.
'''

import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Default colors for multiple methods (ours + baselines)
DEFAULT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",
    "#e377c2",
]

def plot_sliding_window_metrics(exposure_metrics: List[Dict[str, Any]], trigger_metrics: List[Dict[str, Any]], save_path: str):
    '''
    Args:
        exposure_metrics: List[Dict[str, Any]]
        [{
            "exposure_round": i + 1,
            "payload_in_memory_count": payload_in_memory_count,
            "full_metrics": step_metrics
        },
        ...]
        trigger_metrics: List[Dict[str, Any]]
        [{
            "trigger_round": i + 1,
            "payload_in_memory_count": payload_in_memory_count,
            "reload_payload_times": reload_payload_times,
            "exfiltration": exfiltration,
            "command_exec": command_exec,
            "asr_success": exfiltration or command_exec,
            "full_metrics": step_metrics
        },
        ...]
        save_path: str
    Output:
    - Payload count with time
    - Reload times with time
    - Exfiltration ASR with time (scatter plot with time, avg with std)
    - Command Exec ASR with time (scatter plot with time, avg with std)
    - ASR success count so far with time (cumulative sum)
    - Exfiltration success count so far with time (cumulative sum)
    - Command exec success count so far with time (cumulative sum)
    
    All in one figure.
    '''
    pass


def plot_sliding_window_metrics_multi_runs(
    exposure_metrics_list: List[Dict], trigger_metrics_list: List[Dict], 
    save_path: str,
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
    trigger_reload_times_all = np.zeros((n_runs, n_trigger))

    # Collect data from all runs
    for run_idx, (exposure_metrics, trigger_metrics) in enumerate(zip(exposure_metrics_list, trigger_metrics_list)):
        
        for i, m in enumerate(exposure_metrics):
            exposure_persistence_all[run_idx, i] = 1 if m["payload_in_memory_count"] else 0
        
        for i, m in enumerate(trigger_metrics):
            trigger_persistence_all[run_idx, i] = 1 if m["payload_in_memory_count"] else 0
            trigger_exfiltration_all[run_idx, i] = 1 if m.get("exfiltration", False) else 0
            trigger_command_exec_all[run_idx, i] = 1 if m.get("command_exec", False) else 0
            trigger_reload_times_all[run_idx, i] = m.get("reload_payload_times", 0)
    
    # Compute mean and std
    exposure_persistence_mean = np.mean(exposure_persistence_all, axis=0)
    exposure_persistence_std = np.std(exposure_persistence_all, axis=0)
    
    trigger_persistence_mean = np.mean(trigger_persistence_all, axis=0)
    trigger_persistence_std = np.std(trigger_persistence_all, axis=0)
    
    trigger_exfiltration_mean = np.mean(trigger_exfiltration_all, axis=0)
    trigger_exfiltration_std = np.std(trigger_exfiltration_all, axis=0)
    
    trigger_command_exec_mean = np.mean(trigger_command_exec_all, axis=0)
    trigger_command_exec_std = np.std(trigger_command_exec_all, axis=0)

    trigger_reload_times_mean = np.mean(trigger_reload_times_all, axis=0)
    trigger_reload_times_std = np.std(trigger_reload_times_all, axis=0)
    
    # Create round indices
    exposure_rounds = np.arange(1, n_exposure + 1)
    trigger_rounds = np.arange(n_exposure + 1, n_exposure + n_trigger + 1)
    
    # Create figure with four subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))
    
    # Plot 1: Memory Persistence Rate
    for run_idx in range(n_runs):
        ax1.plot(exposure_rounds, exposure_persistence_all[run_idx], 'o-', 
                color='blue', alpha=0.2, linewidth=1, markersize=4)
        ax1.plot(trigger_rounds, trigger_persistence_all[run_idx], 's-', 
                color='orange', alpha=0.2, linewidth=1, markersize=4)
    
    ax1.plot(exposure_rounds, exposure_persistence_mean, 'o-', 
             label=f'Infection Phase (n={n_runs})', color='blue', linewidth=2, markersize=8)
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
    
    # Plot 4: Reload Payload Times
    for run_idx in range(n_runs):
        ax4.plot(trigger_rounds, trigger_reload_times_all[run_idx], 's-', 
                color='green', alpha=0.2, linewidth=1, markersize=4)
    
    ax4.plot(trigger_rounds, trigger_reload_times_mean, 's-', 
             label=f'Reload Times (n={n_runs})', color='green', linewidth=2, markersize=8)
    ax4.fill_between(trigger_rounds, 
                     trigger_reload_times_mean - trigger_reload_times_std,
                     trigger_reload_times_mean + trigger_reload_times_std,
                     color='green', alpha=0.2)
    
    ax4.axvline(x=n_exposure, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Phase Transition')
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Reload Payload Times', fontsize=12)
    ax4.set_title('Payload Reload Times Over Rounds\n(Mean ± Std)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
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
    print(f"  Reload Payload Times: {trigger_reload_times_mean.mean():.2f} ± {trigger_reload_times_std.mean():.2f}")
    print(f"\nMemory Persistence:")
    print(f"  Infection: {exposure_persistence_mean.mean():.2%} ± {exposure_persistence_std.mean():.2%}")
    print(f"  Trigger: {trigger_persistence_mean.mean():.2%} ± {trigger_persistence_std.mean():.2%}")
    print(f"{'='*80}\n")


def _compute_sliding_window_stats(
    exposure_metrics_list: List[List[Dict]],
    trigger_metrics_list: List[List[Dict]],
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    int, int,
]:
    """
    Compute mean and std arrays for one method (multiple runs).

    Returns:
        exposure_rounds, trigger_rounds,
        exfil_mean, exfil_std (cumulative average ASR, trigger phase),
        cmd_mean, cmd_std (cumulative average ASR, trigger phase),
        retention_mean, retention_std (exposure + trigger, 0/1 payload in memory),
        n_exposure, n_trigger
    """
    n_runs = len(exposure_metrics_list)
    n_exposure = len(exposure_metrics_list[0])
    n_trigger = len(trigger_metrics_list[0])

    exposure_persistence_all = np.zeros((n_runs, n_exposure))
    trigger_persistence_all = np.zeros((n_runs, n_trigger))
    trigger_exfiltration_all = np.zeros((n_runs, n_trigger))
    trigger_command_exec_all = np.zeros((n_runs, n_trigger))

    for run_idx, (exposure_metrics, trigger_metrics) in enumerate(
        zip(exposure_metrics_list, trigger_metrics_list)
    ):
        for i, m in enumerate(exposure_metrics):
            exposure_persistence_all[run_idx, i] = (
                1 if m.get("payload_in_memory_count") else 0
            )
        for i, m in enumerate(trigger_metrics):
            trigger_persistence_all[run_idx, i] = (
                1 if m.get("payload_in_memory_count") else 0
            )
            trigger_exfiltration_all[run_idx, i] = (
                1 if m.get("exfiltration") else 0
            )
            trigger_command_exec_all[run_idx, i] = (
                1 if m.get("command_exec") else 0
            )

    # Cumulative average ASR: Y_t = (1/t) * sum_{i=1}^{t} Success_i (per run, then mean±std)
    t_range = np.arange(1, n_trigger + 1, dtype=float)
    exfil_cumavg = np.cumsum(trigger_exfiltration_all, axis=1) / t_range[np.newaxis, :]
    cmd_cumavg = np.cumsum(trigger_command_exec_all, axis=1) / t_range[np.newaxis, :]
    exfil_mean = np.mean(exfil_cumavg, axis=0)
    exfil_std = np.std(exfil_cumavg, axis=0)
    cmd_mean = np.mean(cmd_cumavg, axis=0)
    cmd_std = np.std(cmd_cumavg, axis=0)

    retention_exposure_mean = np.mean(exposure_persistence_all, axis=0)
    retention_exposure_std = np.std(exposure_persistence_all, axis=0)
    retention_trigger_mean = np.mean(trigger_persistence_all, axis=0)
    retention_trigger_std = np.std(trigger_persistence_all, axis=0)

    exposure_rounds = np.arange(1, n_exposure + 1)
    trigger_rounds = np.arange(n_exposure + 1, n_exposure + n_trigger + 1)

    return (
        exposure_rounds,
        trigger_rounds,
        exfil_mean,
        exfil_std,
        cmd_mean,
        cmd_std,
        retention_exposure_mean,
        retention_exposure_std,
        retention_trigger_mean,
        retention_trigger_std,
        n_exposure,
        n_trigger,
    )


def _save_fig_sw(fig, save_dir: str, base_name: str) -> None:
    """Save figure as PDF only."""
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, base_name + ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    print(f"[Sliding window plot saved to {pdf_path}]")


def plot_sliding_window_asr_and_retention(
    series: List[Tuple[str, List[List[Dict]], List[List[Dict]]]],
    save_dir: Optional[str] = None,
    filename_prefix: str = "sliding_window",
    colors: Optional[List[str]] = None,
) -> None:
    """
    Plot two figures for the sliding window scenario (same layout as RAG):
    1. ASR: one figure with two subfigures — (a) Data exfiltration ASR, (b) Command execution ASR.
    2. Payload retention: one standalone figure.

    Args:
        series: List of (label, exposure_metrics_list, trigger_metrics_list).
        save_dir: Directory to save figures. If None, only plt.show().
        filename_prefix: Prefix for output files (e.g. "sliding_window" -> sliding_window_asr.pdf, sliding_window_payload_retention.pdf).
        colors: Optional list of colors (one per series). Uses DEFAULT_COLORS if None.
    """
    if not series:
        raise ValueError("series must be non-empty")

    n_exposure_ref = len(series[0][1][0])
    n_trigger_ref = len(series[0][2][0])

    colors = colors or DEFAULT_COLORS
    if len(colors) < len(series):
        colors = colors + [plt.cm.tab10(i) for i in range(len(series) - len(colors))]

    # --- Figure 1: ASR (two subfigures) ---
    fig1, (ax_exfil, ax_cmd) = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (label, exp_list, trig_list) in enumerate(series):
        (
            exposure_rounds,
            trigger_rounds,
            exfil_mean,
            exfil_std,
            cmd_mean,
            cmd_std,
            _,
            _,
            _,
            _,
            n_exp,
            n_trig,
        ) = _compute_sliding_window_stats(exp_list, trig_list)
        c = colors[idx % len(colors)]
        # Exfiltration ASR
        ax_exfil.plot(
            trigger_rounds,
            exfil_mean,
            "o-",
            color=c,
            linewidth=2,
            markersize=6,
            label=label,
        )
        ax_exfil.fill_between(
            trigger_rounds,
            exfil_mean - exfil_std,
            exfil_mean + exfil_std,
            color=c,
            alpha=0.2,
        )
        # Command execution ASR
        ax_cmd.plot(
            trigger_rounds,
            cmd_mean,
            "s-",
            color=c,
            linewidth=2,
            markersize=6,
            label=label,
        )
        ax_cmd.fill_between(
            trigger_rounds,
            cmd_mean - cmd_std,
            cmd_mean + cmd_std,
            color=c,
            alpha=0.2,
        )

    for ax in (ax_exfil, ax_cmd):
        ax.axvline(x=n_exposure_ref, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("累计平均 ASR", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    ax_exfil.set_title("Data Exfiltration ASR", fontsize=14, fontweight="bold")
    ax_cmd.set_title("Command Execution ASR", fontsize=14, fontweight="bold")
    fig1.suptitle("Sliding Window: Cumulative Average ASR", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_dir:
        _save_fig_sw(fig1, save_dir, f"{filename_prefix}_asr")
    plt.show()
    plt.close(fig1)

    # --- Figure 2: Payload retention (standalone) ---
    fig2, ax_ret = plt.subplots(1, 1, figsize=(10, 5))

    for idx, (label, exp_list, trig_list) in enumerate(series):
        (
            exposure_rounds,
            trigger_rounds,
            _,
            _,
            _,
            _,
            ret_exp_mean,
            ret_exp_std,
            ret_trig_mean,
            ret_trig_std,
            n_exp,
            n_trig,
        ) = _compute_sliding_window_stats(exp_list, trig_list)
        c = colors[idx % len(colors)]
        all_rounds = np.concatenate([exposure_rounds, trigger_rounds])
        ret_mean = np.concatenate([ret_exp_mean, ret_trig_mean])
        ret_std = np.concatenate([ret_exp_std, ret_trig_std])
        ax_ret.plot(
            all_rounds,
            ret_mean,
            "o-",
            color=c,
            linewidth=2,
            markersize=5,
            label=label,
        )
        ax_ret.fill_between(
            all_rounds,
            ret_mean - ret_std,
            ret_mean + ret_std,
            color=c,
            alpha=0.2,
        )

    ax_ret.axvline(
        x=n_exposure_ref,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Phase transition",
    )
    n_trigger_ref = len(series[0][2][0])
    ymin, ymax = ax_ret.get_ylim()
    ax_ret.text(0.3 * (1 + n_exposure_ref), ymax * 0.7, "Infection", ha="center", fontsize=11, fontweight="bold")
    ax_ret.text(n_exposure_ref + 0.5 * n_trigger_ref, ymax * 0.7, "Trigger", ha="center", fontsize=11, fontweight="bold")
    ax_ret.set_xlabel("Round", fontsize=12)
    ax_ret.set_ylabel("Payload in Memory (rate)", fontsize=12)
    ax_ret.set_title("Sliding Window: Payload Retention", fontsize=14, fontweight="bold")
    ax_ret.set_ylim(-0.05, 1.05)
    ax_ret.legend(fontsize=10)
    ax_ret.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        _save_fig_sw(fig2, save_dir, f"{filename_prefix}_payload_retention")
    plt.show()
    plt.close(fig2)
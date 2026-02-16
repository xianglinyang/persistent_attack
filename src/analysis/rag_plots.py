'''
File to plot the metrics from RAG agent attacks.
'''

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional

# Colors for multiple methods (ours + baselines)
RAG_PLOT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
]

# def plot_exposure_metrics(exposure_metrics, save_path):
#     '''
#     Args:
#         exposure_metrics: List[Dict[str, Any]]
#         [{
#             "exposure_round": current_round (time),
#             "rag_payload_count": payload_count,
#             "full_metrics": step_metrics
#         },
#         ...
#         ]
#         save_path: str

#     Subplots:
#     - Payload count with time
#     '''
#     pass


# def plot_trigger_metrics(trigger_metrics, save_path):
#     '''
#     Args:
#         trigger_metrics: List[Dict[str, Any]]
#         [{
#             "trigger_round": i(trigger round index),
#             "exposure_round": exposure_round,
#             "rag_payload_count": rag_payload_cnt,
#             "recall@10": recall_10,
#             "recall@50": recall_50, 
#             "recall@100": recall_100,
#             "exfiltration": exfiltration,
#             "command_exec": command_exec,
#             "reload_payload_times": reload_payload_times,
#             "asr_success": asr_success,
#             "full_metrics": step_metrics,
#         },
#         ...]
#         save_path: str

#     Subplots:
#     - Payload count with (line plot with time, avg with std)
#     - Recall@10, Recall@50, Recall@100 with time (Three lines) Avg with std
#     - Exfiltration ASR with time (scatter plot with time, avg with std)
#     - Command Exec ASR with time (scatter plot with time, avg with std)
#     - Reload Times with time (line plot with time, avg with std)
    
#     - Exfiltration success count so far with time (line plot with time, avg with std) (cumulative sum)
#     - Command exec success count so far with time (line plot with time, avg with std) (cumulative sum)
#     All in one figure.
#     '''
#     pass


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_exposure_metrics(exposure_metrics: List[Dict[str, Any]], save_path: str):
    df = pd.DataFrame(exposure_metrics)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Simple line plot for payload count over time
    sns.lineplot(data=df, x="exposure_round", y="rag_payload_count", marker='o')
    
    plt.title("RAG Payload Count over Infection Rounds")
    plt.xlabel("Infection Round")
    plt.ylabel("Payload Count")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def plot_trigger_metrics(trigger_metrics: List[Dict[str, Any]], save_path: str):
    all_dfs = []
    for i, run_data in enumerate(trigger_metrics):
        df_run = pd.DataFrame(run_data)
        df_run['run_id'] = i + 1
        all_dfs.append(df_run)
    
    # sort by trigger_round and run_id
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values(by=["run_id", "trigger_round"])

    # Create a figure with 5 subplots (arranged in a single row)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    sns.set_style("whitegrid")

    # 1. Payload Count (Line plot, avg with std)
    sns.lineplot(ax=axes[0], data=df, x="trigger_round", y="rag_payload_count", errorbar='sd')
    axes[0].set_title("Payload Count over Time")

    # 2. Recall@K (Three lines)
    # Melt the dataframe to make it "long-form" for easier plotting with Seaborn
    recall_df = df.melt(id_vars=['trigger_round'], value_vars=['recall@10', 'recall@50', 'recall@100'], 
                        var_name='Metric', value_name='Recall')
    sns.lineplot(ax=axes[1], data=recall_df, x="trigger_round", y="Recall", hue='Metric', errorbar='sd')
    axes[1].set_title("Recall Performance (10, 50, 100)")

    # 3. Exfiltration ASR (Scatter + Regression/Avg line)
    sns.scatterplot(ax=axes[2], data=df, x="trigger_round", y="exfiltration", alpha=0.5)
    sns.lineplot(ax=axes[2], data=df, x="trigger_round", y="exfiltration", errorbar='sd', color='red')
    axes[2].set_title("Exfiltration ASR over Time")

    # 4. Command Exec ASR (Scatter + Regression/Avg line)
    sns.scatterplot(ax=axes[3], data=df, x="trigger_round", y="command_exec", alpha=0.5)
    sns.lineplot(ax=axes[3], data=df, x="trigger_round", y="command_exec", errorbar='sd', color='green')
    axes[3].set_title("Command Exec ASR over Time")

    # 5. Reload Times
    sns.lineplot(ax=axes[4], data=df, x="trigger_round", y="reload_payload_times", errorbar='sd')
    axes[4].set_title("Reload Times over Time")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _compute_rag_stats(
    exposure_metrics_list: List[List[Dict]],
    trigger_metrics_list: List[List[Dict]],
):
    """
    Compute mean and std arrays for one RAG method (multiple runs).
    ASR (exfil, cmd) are cumulative average: Y_t = (1/t) * sum_{i=1}^{t} Success_i.
    Returns trigger_rounds, exfil_cumavg_mean, exfil_cumavg_std, cmd_cumavg_mean, cmd_cumavg_std,
    recall10/50/100 mean&std, payload_mean, payload_std (trigger phase).
    """
    n_runs = len(trigger_metrics_list)
    n_trigger = len(trigger_metrics_list[0])

    exfil_all = np.zeros((n_runs, n_trigger))
    cmd_all = np.zeros((n_runs, n_trigger))
    recall10_all = np.zeros((n_runs, n_trigger))
    recall50_all = np.zeros((n_runs, n_trigger))
    recall100_all = np.zeros((n_runs, n_trigger))
    payload_all = np.zeros((n_runs, n_trigger))

    for run_idx, trigger_metrics in enumerate(trigger_metrics_list):
        for i, m in enumerate(trigger_metrics):
            exfil_all[run_idx, i] = 1 if m.get("exfiltration") else 0
            cmd_all[run_idx, i] = 1 if m.get("command_exec") else 0
            recall10_all[run_idx, i] = m.get("recall@10", 0)
            recall50_all[run_idx, i] = m.get("recall@50", 0)
            recall100_all[run_idx, i] = m.get("recall@100", 0)
            payload_all[run_idx, i] = m.get("rag_payload_count", 0)

    # Cumulative average ASR: Y_t = (1/t) * sum_{i=1}^{t} Success_i (per run, then mean±std)
    t_range = np.arange(1, n_trigger + 1, dtype=float)
    exfil_cumavg = np.cumsum(exfil_all, axis=1) / t_range[np.newaxis, :]
    cmd_cumavg = np.cumsum(cmd_all, axis=1) / t_range[np.newaxis, :]
    exfil_cumavg_mean = np.mean(exfil_cumavg, axis=0)
    exfil_cumavg_std = np.std(exfil_cumavg, axis=0)
    cmd_cumavg_mean = np.mean(cmd_cumavg, axis=0)
    cmd_cumavg_std = np.std(cmd_cumavg, axis=0)

    trigger_rounds = np.arange(1, n_trigger + 1)
    return (
        trigger_rounds,
        exfil_cumavg_mean, exfil_cumavg_std,
        cmd_cumavg_mean, cmd_cumavg_std,
        np.mean(recall10_all, axis=0), np.std(recall10_all, axis=0),
        np.mean(recall50_all, axis=0), np.std(recall50_all, axis=0),
        np.mean(recall100_all, axis=0), np.std(recall100_all, axis=0),
        np.mean(payload_all, axis=0), np.std(payload_all, axis=0),
    )


def _save_fig(fig, save_dir: str, base_name: str) -> None:
    """Save figure as PDF only."""
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, base_name + ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    print(f"[RAG plot saved to {pdf_path}]")


def plot_rag_asr_and_retention(
    series: List[Tuple[str, List[List[Dict]], List[List[Dict]]]],
    save_dir: Optional[str] = None,
    filename_prefix: str = "rag",
    colors: Optional[List[str]] = None,
) -> None:
    """
    Plot RAG metrics as three figures:
    1. ASR: one figure with two subfigures (exfiltration + command exec)
    2. Recall: one figure with three box-plot subplots (检索分布箱线图) — Recall@10, @50, @100, each showing distribution by method
    3. Payload count (standalone)

    series: List of (label, exposure_metrics_list, trigger_metrics_list).
    Saves PDF for each when save_dir is set.
    """
    if not series:
        raise ValueError("series must be non-empty")

    colors = colors or RAG_PLOT_COLORS
    if len(colors) < len(series):
        colors = list(colors) + [plt.cm.tab10(i) for i in range(len(series) - len(colors))]

    def _ax_style(ax, ylabel: str, title: str):
        ax.set_xlabel("Trigger Round", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight="bold")

    # --- 1. ASR: one figure with two subfigures (exfiltration + command exec) ---
    fig_asr, (ax_exfil, ax_cmd) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (label, _exp_list, trig_list) in enumerate(series):
        trigger_rounds, exfil_mean, exfil_std, cmd_mean, cmd_std, *_ = _compute_rag_stats(_exp_list, trig_list)[:5]
        c = colors[idx % len(colors)]
        ax_exfil.plot(trigger_rounds, exfil_mean, "o-", color=c, linewidth=2, markersize=6, label=label)
        ax_exfil.fill_between(trigger_rounds, exfil_mean - exfil_std, exfil_mean + exfil_std, color=c, alpha=0.2)
        ax_cmd.plot(trigger_rounds, cmd_mean, "s-", color=c, linewidth=2, markersize=6, label=label)
        ax_cmd.fill_between(trigger_rounds, cmd_mean - cmd_std, cmd_mean + cmd_std, color=c, alpha=0.2)
    for ax in (ax_exfil, ax_cmd):
        ax.set_ylim(-0.05, 1.05)
    _ax_style(ax_exfil, "Cumulative Average ASR", "Data Exfiltration ASR")
    _ax_style(ax_cmd, "Cumulative Average ASR", "Command Execution ASR")
    fig_asr.suptitle("RAG: Cumulative Average ASR", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_dir:
        _save_fig(fig_asr, save_dir, f"{filename_prefix}_asr")
    plt.show()
    plt.close(fig_asr)

    # --- 2. Recall: Grouped Bar Chart — X=K (Top-10/50/100), Y=Average Retrieval Count, Hue=Method, Error=SE ---
    recall10_by_method = []
    recall50_by_method = []
    recall100_by_method = []
    method_labels = []
    for label, _exp_list, trig_list in series:
        r10_flat, r50_flat, r100_flat = [], [], []
        for trigger_metrics in trig_list:
            for m in trigger_metrics:
                r10_flat.append(m.get("recall@10", 0))
                r50_flat.append(m.get("recall@50", 0))
                r100_flat.append(m.get("recall@100", 0))
        recall10_by_method.append(r10_flat)
        recall50_by_method.append(r50_flat)
        recall100_by_method.append(r100_flat)
        method_labels.append(label)

    n_methods = len(method_labels)
    # Mean and Standard Error (SE = std / sqrt(n)) per method per K
    def mean_and_se(data_list):
        means = [np.mean(d) if len(d) else 0 for d in data_list]
        ses = [np.std(d) / np.sqrt(len(d)) if len(d) > 1 else 0 for d in data_list]
        return means, ses

    m10, se10 = mean_and_se(recall10_by_method)
    m50, se50 = mean_and_se(recall50_by_method)
    m100, se100 = mean_and_se(recall100_by_method)

    group_centers = np.arange(3)  # Top-10, Top-50, Top-100
    total_width = 0.8
    bar_width = total_width / n_methods

    fig_recall, ax = plt.subplots(1, 1, figsize=(8, 5))
    for method_idx in range(n_methods):
        offset = (method_idx - (n_methods - 1) / 2) * bar_width
        x_pos = group_centers + offset
        y_vals = [m10[method_idx], m50[method_idx], m100[method_idx]]
        y_err = [se10[method_idx], se50[method_idx], se100[method_idx]]
        ax.bar(
            x_pos,
            y_vals,
            bar_width,
            yerr=y_err,
            capsize=3,
            label=method_labels[method_idx],
            color=colors[method_idx % len(colors)],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(["Top-10", "Top-50", "Top-100"], fontsize=12)
    ax.set_ylabel("Average Retrieval Count", fontsize=12)
    ax.set_xlabel("K (Retrieval Top-K)", fontsize=12)
    ax.set_title("RAG: Retrieval Count by K (Mean ± SE)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_dir:
        _save_fig(fig_recall, save_dir, f"{filename_prefix}_recall_bar")
    plt.show()
    plt.close(fig_recall)

    # --- 3. Payload count: infection + trigger in one figure ---
    fig_payload, ax_payload = plt.subplots(1, 1, figsize=(8, 4))
    n_exposure_ref = len(series[0][1][0])
    for idx, (label, exp_list, trig_list) in enumerate(series):
        stats = _compute_rag_stats(exp_list, trig_list)
        n_trigger = len(trig_list[0])
        # Exposure phase: use first run (exposure is same across runs)
        exposure_metrics = exp_list[0]
        exposure_rounds = np.arange(1, len(exposure_metrics) + 1)
        exposure_payload = np.array([m.get("rag_payload_count", 0) for m in exposure_metrics])
        # Trigger phase: mean ± std
        trigger_rounds_global = np.arange(len(exposure_metrics) + 1, len(exposure_metrics) + n_trigger + 1)
        payload_mean, payload_std = stats[11], stats[12]
        c = colors[idx % len(colors)]
        ax_payload.plot(exposure_rounds, exposure_payload, "o-", color=c, linewidth=2, markersize=5, label=label)
        ax_payload.plot(trigger_rounds_global, payload_mean, "s-", color=c, linewidth=2, markersize=5)
        ax_payload.fill_between(trigger_rounds_global, payload_mean - payload_std, payload_mean + payload_std, color=c, alpha=0.2)
    ax_payload.axvline(x=n_exposure_ref, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="Phase transition")
    n_trigger_ref = len(series[0][2][0])
    ymin, ymax = ax_payload.get_ylim()
    ax_payload.text(0.5 * (1 + n_exposure_ref), ymax * 0.85, "Infection", ha="center", fontsize=11, fontweight="bold")
    ax_payload.text(n_exposure_ref + 0.8 * n_trigger_ref, ymax * 0.85, "Trigger", ha="center", fontsize=11, fontweight="bold")
    ax_payload.set_xlabel("Round", fontsize=12)
    ax_payload.set_ylabel("Payload Count", fontsize=12)
    ax_payload.legend(fontsize=10)
    ax_payload.grid(True, alpha=0.3)
    ax_payload.set_title("RAG: Payload Count (Infection + Trigger)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        _save_fig(fig_payload, save_dir, f"{filename_prefix}_payload_count")
    plt.show()
    plt.close(fig_payload)


def plot_rag_evolution_asr_grouped(
    series: List[Tuple[str, List[List[Dict]], List[List[Dict]]]],
    save_dir: Optional[str] = None,
    filename: str = "rag_evolution_asr.pdf",
    evolution_labels: Optional[List[str]] = None,
) -> None:
    """
    Grouped bar chart: Evolution Strategy (X) vs ASR (Y), Hue = Attack Task (Exfiltration / Command Execution).
    Use for comparing Raw vs Reflection vs Experience under Data Exfiltration and Command Execution.
    series: list of (label, exposure_metrics_list, trigger_metrics_list), length 3 (raw, reflection, experience).
    """
    if len(series) != 3:
        raise ValueError("plot_rag_evolution_asr_grouped expects exactly 3 series (raw, reflection, experience).")
    if evolution_labels is None:
        evolution_labels = ["Raw History", "Verbal Reflection", "Refined Experience"]
    if len(evolution_labels) != 3:
        evolution_labels = [s[0] for s in series]

    # For each evolution strategy: flatten exfil and cmd (0/1), compute mean and SE
    exfil_means, exfil_ses = [], []
    cmd_means, cmd_ses = [], []
    for _label, _exp_list, trig_list in series:
        exfil_flat, cmd_flat = [], []
        for trigger_metrics in trig_list:
            for m in trigger_metrics:
                exfil_flat.append(1 if m.get("exfiltration") else 0)
                cmd_flat.append(1 if m.get("command_exec") else 0)
        n = len(exfil_flat)
        exfil_means.append(np.mean(exfil_flat) if n else 0)
        exfil_ses.append(np.std(exfil_flat) / np.sqrt(n) if n > 1 else 0)
        cmd_means.append(np.mean(cmd_flat) if n else 0)
        cmd_ses.append(np.std(cmd_flat) / np.sqrt(n) if n > 1 else 0)

    # Grouped bar: X = 3 evolution strategies, 2 bars per group (Exfiltration, Command Execution)
    group_centers = np.arange(3)
    bar_width = 0.35
    x_exfil = group_centers - bar_width / 2
    x_cmd = group_centers + bar_width / 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(
        x_exfil,
        exfil_means,
        bar_width,
        yerr=exfil_ses,
        capsize=4,
        label="Data Exfiltration",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x_cmd,
        cmd_means,
        bar_width,
        yerr=cmd_ses,
        capsize=4,
        label="Command Execution",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(evolution_labels, fontsize=11)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=12)
    ax.set_xlabel("Evolution Strategy", fontsize=12)
    ax.set_title("ASR by Evolution Strategy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)
        fig.savefig(out_path, bbox_inches="tight", format="pdf")
        print(f"[RAG evolution ASR plot saved to {out_path}]")
    plt.show()
    plt.close(fig)


def plot_rag_evolution_payload_grouped(
    series: List[Tuple[str, List[List[Dict]], List[List[Dict]]]],
    save_dir: Optional[str] = None,
    filename: str = "rag_evolution_payload_count.pdf",
    evolution_labels: Optional[List[str]] = None,
) -> None:
    """
    Grouped bar: Evolution Strategy (X) vs mean Payload Count (Y). Uses trigger session rag_payload_count.
    series: list of (label, exposure_metrics_list, trigger_metrics_list), length 3 (raw, reflection, experience).
    """
    if len(series) != 3:
        raise ValueError("plot_rag_evolution_payload_grouped expects exactly 3 series (raw, reflection, experience).")
    if evolution_labels is None:
        evolution_labels = [s[0] for s in series]
    if len(evolution_labels) != 3:
        evolution_labels = [s[0] for s in series]

    payload_means, payload_ses = [], []
    for _label, _exp_list, trig_list in series:
        flat = []
        for trigger_metrics in trig_list:
            for m in trigger_metrics:
                flat.append(m.get("rag_payload_count", 0))
        n = len(flat)
        payload_means.append(np.mean(flat) if n else 0)
        payload_ses.append(np.std(flat) / np.sqrt(n) if n > 1 else 0)

    group_centers = np.arange(3)
    bar_width = 0.6
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(
        group_centers,
        payload_means,
        bar_width,
        yerr=payload_ses,
        capsize=4,
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.5,
        label="Payload Count (mean)",
    )
    ax.set_xticks(group_centers)
    ax.set_xticklabels(evolution_labels, fontsize=11)
    ax.set_ylabel("Payload Count (mean)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlabel("Evolution Strategy", fontsize=12)
    ax.set_title("Payload Count by Evolution Strategy (Trigger Session)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)
        fig.savefig(out_path, bbox_inches="tight", format="pdf")
        print(f"[RAG evolution payload count plot saved to {out_path}]")
    plt.show()
    plt.close(fig)


def _asr_mean_and_se_from_trigger(trigger_metrics_list: List[List[Dict]]) -> Tuple[float, float]:
    """Compute ASR = mean(success) where success = exfiltration OR command_exec; return (mean, SE)."""
    flat = []
    for trigger_metrics in trigger_metrics_list:
        for m in trigger_metrics:
            flat.append(1 if (m.get("exfiltration") or m.get("command_exec")) else 0)
    n = len(flat)
    if n == 0:
        return 0.0, 0.0
    mean = np.mean(flat)
    se = np.std(flat) / np.sqrt(n) if n > 1 else 0.0
    return float(mean), float(se)


def _exfil_cmd_mean_se_from_trigger(
    trigger_metrics_list: List[List[Dict]],
) -> Tuple[float, float, float, float]:
    """Return (exfil_mean, exfil_se, cmd_mean, cmd_se)."""
    exfil_flat, cmd_flat = [], []
    for trigger_metrics in trigger_metrics_list:
        for m in trigger_metrics:
            exfil_flat.append(1 if m.get("exfiltration") else 0)
            cmd_flat.append(1 if m.get("command_exec") else 0)
    n = len(exfil_flat)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    exfil_mean = np.mean(exfil_flat)
    exfil_se = np.std(exfil_flat) / np.sqrt(n) if n > 1 else 0.0
    cmd_mean = np.mean(cmd_flat)
    cmd_se = np.std(cmd_flat) / np.sqrt(n) if n > 1 else 0.0
    return float(exfil_mean), float(exfil_se), float(cmd_mean), float(cmd_se)


def plot_defense_asr_grouped(
    defense_data: List[Tuple[str, List[Tuple[str, float, float]]]],
    save_dir: Optional[str] = None,
    filename: str = "defense_asr_grouped.pdf",
    scenario: str = "RAG",
) -> None:
    """
    Grouped bar: X = Instruction guard (raw, raw+sandwich, ...), Y = ASR, Hue = Data Exfiltration vs Command Execution.
    Works for both RAG and Sliding Window trigger metrics (same structure).
    defense_data: list of (defense_name, [(metric_label, mean, se), ...]) with 2 metrics: Data Exfiltration, Command Execution.
    scenario: "RAG" or "Sliding Window" (for title only).
    """
    if not defense_data:
        raise ValueError("defense_data must be non-empty")
    defense_names = [d[0] for d in defense_data]
    metric_labels = ["Data Exfiltration", "Command Execution"]
    n_defenses = len(defense_names)
    bar_width = 0.35
    group_centers = np.arange(n_defenses)

    fig, ax = plt.subplots(1, 1, figsize=(max(8, n_defenses * 1.5), 5))
    for metric_idx, metric_label in enumerate(metric_labels):
        means = []
        ses = []
        for def_name, methods in defense_data:
            found = next((m for m in methods if m[0] == metric_label), None)
            if found:
                means.append(found[1])
                ses.append(found[2])
            else:
                means.append(0.0)
                ses.append(0.0)
        offset = (metric_idx - 0.5) * bar_width
        x_pos = group_centers + offset
        color = "#1f77b4" if metric_idx == 0 else "#ff7f0e"
        ax.bar(
            x_pos,
            means,
            bar_width,
            yerr=ses,
            capsize=4,
            label=metric_label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(defense_names, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=12)
    ax.set_xlabel("Instruction Guard", fontsize=12)
    ax.set_title(f"ASR by Instruction Defense ({scenario})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)
        fig.savefig(out_path, bbox_inches="tight", format="pdf")
        print(f"[Instruction defense ASR plot saved to {out_path}]")
    plt.show()
    plt.close(fig)
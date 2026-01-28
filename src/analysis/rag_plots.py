'''
File to plot the metrics from RAG agent attacks.
'''

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

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
    
    plt.title("RAG Payload Count over Exposure Rounds")
    plt.xlabel("Exposure Round")
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

    df['cum_exfiltration'] = df['exfiltration'].cumsum()
    df['cum_command_exec'] = df['command_exec'].cumsum()

    # Create a figure with 7 subplots (arranged 4x2 or similar)
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
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

    # 6. Cumulative Exfiltration
    sns.lineplot(ax=axes[5], data=df, x="trigger_round", y="cum_exfiltration", color='purple')
    axes[5].set_title("Cumulative Exfiltration Successes")

    # 7. Cumulative Command Exec
    sns.lineplot(ax=axes[6], data=df, x="trigger_round", y="cum_command_exec", color='orange')
    axes[6].set_title("Cumulative Command Exec Successes")

    # Remove the empty 8th subplot
    fig.delaxes(axes[7])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
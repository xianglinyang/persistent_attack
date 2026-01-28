'''
File to save the metrics to a JSON file.
'''

# def save_exposure_metrics(exposure_metrics, logs, save_path):

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
#         logs: Dict[str, Any]
#         save_path: str
#     '''
#     pass

# def save_trigger_metrics(trigger_metrics, logs, save_path):
#     '''
#     Args:
#         trigger_metrics: List[Dict[str, Any]]
#         [{
#             "trigger_round": i(trigger round index),
#             "exposure_round": exposure_round (time),
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
#         ...
#         ]
#         logs: Dict[str, Any]
#         save_path: str
#     '''
#     pass

import json
import pandas as pd
from pathlib import Path

def save_exposure_metrics(exposure_metrics, logs, save_path):
    """
    Saves exposure metrics and logs to a JSON file and a CSV for the flat data.
    """
    output_dir = Path(save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save complete data (including logs and full_metrics) to JSON
    combined_data = {
        "logs": logs,
        "metrics": exposure_metrics
    }
    with open(save_path, 'w') as f:
        json.dump(combined_data, f, indent=4)

    # 2. Save flat metrics to CSV for quick analysis
    # We exclude 'full_metrics' from the CSV to keep it clean
    flat_metrics = []
    for entry in exposure_metrics:
        flat_entry = {k: v for k, v in entry.items() if k != 'full_metrics'}
        flat_metrics.append(flat_entry)
    
    csv_path = save_path.replace('.json', '.csv')
    pd.DataFrame(flat_metrics).to_csv(csv_path, index=False)
    print(f"Exposure metrics saved to {save_path} and {csv_path}")


def save_trigger_metrics(trigger_metrics, logs, save_path):
    """
    Saves trigger metrics and logs to a JSON file and a CSV.
    """
    output_dir = Path(save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save complete data to JSON
    combined_data = {
        "logs": logs,
        "metrics": trigger_metrics
    }
    with open(save_path, 'w') as f:
        json.dump(combined_data, f, indent=4)

    # 2. Save flat metrics to CSV
    # Extract all keys except the nested 'full_metrics'
    flat_metrics = []
    for entry in trigger_metrics:
        flat_entry = {k: v for k, v in entry.items() if k != 'full_metrics'}
        flat_metrics.append(flat_entry)

    csv_path = save_path.replace('.json', '.csv')
    pd.DataFrame(flat_metrics).to_csv(csv_path, index=False)
    print(f"Trigger metrics saved to {save_path} and {csv_path}")
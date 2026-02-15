'''
Script to regenerate plots from saved metrics JSON files.

Supports both RAG and Sliding Window attack metrics.

Usage:
    # RAG metrics
    python -m src.analysis.plot_from_saved --metrics_path /path/to/metrics.json --output_path /path/to/output.png --plot_type [exposure|trigger]
    
    # Sliding Window metrics
    python -m src.analysis.plot_from_saved --metrics_path /path/to/metrics.json --output_path /path/to/output.png --plot_type sliding_window
    
Or use the convenience function:
    python -m src.analysis.plot_from_saved --save_dir /path/to/save_dir --model_name google/gemini-2.5-flash
'''

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.analysis.rag_plots import (
    plot_exposure_metrics,
    plot_trigger_metrics,
    plot_rag_asr_and_retention,
    plot_rag_evolution_asr_grouped,
    plot_rag_evolution_payload_grouped,
    plot_defense_asr_grouped,
    _asr_mean_and_se_from_trigger,
    _exfil_cmd_mean_se_from_trigger,
)
from src.analysis.sliding_window_plots import (
    plot_sliding_window_metrics_multi_runs,
    plot_sliding_window_asr_and_retention,
)


def load_metrics_from_json(json_path: str) -> tuple:
    """
    Load metrics from a saved JSON file.
    
    Args:
        json_path: Path to the JSON file containing metrics
        
    Returns:
        tuple: (metrics_list, logs_dict)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metrics = data.get("metrics", [])
    logs = data.get("logs", {})
    
    return metrics, logs


def plot_from_saved_exposure(metrics_path: str, output_path: str) -> None:
    """
    Load exposure metrics from JSON and generate plot.
    
    Args:
        metrics_path: Path to exposure_metrics.json
        output_path: Path to save the output plot
    """
    print(f"Loading exposure metrics from: {metrics_path}")
    metrics, logs = load_metrics_from_json(metrics_path)
    
    if not metrics:
        print("Warning: No metrics found in file")
        return
    
    print(f"Found {len(metrics)} exposure rounds")
    print(f"Generating plot at: {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plot_exposure_metrics(metrics, output_path)
    print("✅ Exposure plot generated successfully")


def plot_from_saved_trigger(metrics_path: str, output_path: str) -> None:
    """
    Load trigger metrics from JSON and generate plot.
    
    Args:
        metrics_path: Path to trigger_metrics.json
        output_path: Path to save the output plot
    """
    print(f"Loading trigger metrics from: {metrics_path}")
    metrics, logs = load_metrics_from_json(metrics_path)
    
    if not metrics:
        print("Warning: No metrics found in file")
        return
    
    # Calculate total trigger rounds
    total_rounds = sum(len(batch) for batch in metrics)
    print(f"Found {len(metrics)} trigger batches with {total_rounds} total rounds")
    print(f"Generating plot at: {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plot_trigger_metrics(metrics, output_path)
    print("✅ Trigger plot generated successfully")


def plot_from_saved_sliding_window(exposure_path: str, trigger_path: str, output_path: str) -> None:
    """
    Load sliding window metrics from JSON files and generate plot.
    
    Args:
        exposure_path: Path to exposure metrics JSON (e.g., metrics_exposure_*.json)
        trigger_path: Path to trigger metrics JSON (e.g., metrics_trigger_*.json)
        output_path: Path to save the output plot
    """
    print(f"Loading sliding window exposure metrics from: {exposure_path}")
    exposure_metrics, exposure_logs = load_metrics_from_json(exposure_path)
    
    print(f"Loading sliding window trigger metrics from: {trigger_path}")
    trigger_metrics, trigger_logs = load_metrics_from_json(trigger_path)
    
    if not exposure_metrics:
        print("Warning: No exposure metrics found")
        return
    
    if not trigger_metrics:
        print("Warning: No trigger metrics found")
        return
    
    print(f"Found {len(exposure_metrics)} exposure rounds")
    print(f"Found {len(trigger_metrics)} trigger batches")
    print(f"Generating plot at: {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create list of exposure metrics (one per run)
    # For sliding window, each run uses the same exposure metrics
    exposure_metrics_list = [exposure_metrics for _ in range(len(trigger_metrics))]
    
    plot_sliding_window_metrics_multi_runs(
        exposure_metrics_list=exposure_metrics_list,
        trigger_metrics_list=trigger_metrics,
        save_path=output_path
    )
    print("✅ Sliding window plot generated successfully")


def plot_sliding_window_comparison(
    series_specs: List[tuple],
    save_dir: str,
    filename_prefix: str = "sliding_window",
) -> None:
    """
    Load our results and baselines' results, then plot two figures (same as RAG layout):
    1. ASR: one figure with two subfigures (exfiltration + command exec).
    2. Payload retention: one standalone figure.
    Saves PDF for each.

    Args:
        series_specs: List of (label, exposure_json_path, trigger_json_path).
        save_dir: Directory to save the figures.
        filename_prefix: Prefix for output files (e.g. "sliding_window" -> sliding_window_asr.pdf, sliding_window_payload_retention.pdf).
    """
    series = []
    for label, exposure_path, trigger_path in series_specs:
        exposure_metrics, _ = load_metrics_from_json(exposure_path)
        trigger_metrics, _ = load_metrics_from_json(trigger_path)
        if not exposure_metrics:
            print(f"Warning: No exposure metrics in {exposure_path}, skipping {label}")
            continue
        if not trigger_metrics:
            print(f"Warning: No trigger metrics in {trigger_path}, skipping {label}")
            continue
        # trigger_metrics from JSON is list of runs (each run = list of round dicts)
        # exposure_metrics is a single list of round dicts; we replicate per run
        n_runs = len(trigger_metrics)
        exposure_metrics_list = [exposure_metrics for _ in range(n_runs)]
        series.append((label, exposure_metrics_list, trigger_metrics))
        print(f"Loaded {label}: {len(exposure_metrics)} exposure rounds, {n_runs} trigger runs")
    if not series:
        print("Error: No valid series loaded.")
        return
    plot_sliding_window_asr_and_retention(
        series,
        save_dir=save_dir,
        filename_prefix=filename_prefix,
    )
    print("✅ Sliding window comparison plots generated successfully")


def plot_rag_comparison(
    series_specs: List[tuple],
    save_dir: str,
    filename_prefix: str = "rag",
) -> None:
    """
    Load our results and baselines' results for RAG, then plot three figures:
    ASR (exfiltration + command exec), Recall box plot (检索分布箱线图), payload count.
    Saves PDF for each.

    Args:
        series_specs: List of (label, exposure_json_path, trigger_json_path).
        save_dir: Directory to save the figures.
        filename_prefix: Prefix for output files (e.g. "rag" -> rag_asr.pdf, rag_recall_box.pdf, ...).
    """
    series = []
    for label, exposure_path, trigger_path in series_specs:
        exposure_metrics, _ = load_metrics_from_json(exposure_path)
        trigger_metrics, _ = load_metrics_from_json(trigger_path)
        if not exposure_metrics:
            print(f"Warning: No exposure metrics in {exposure_path}, skipping {label}")
            continue
        if not trigger_metrics:
            print(f"Warning: No trigger metrics in {trigger_path}, skipping {label}")
            continue
        n_runs = len(trigger_metrics)
        exposure_metrics_list = [exposure_metrics for _ in range(n_runs)]
        series.append((label, exposure_metrics_list, trigger_metrics))
        print(f"Loaded RAG {label}: {len(exposure_metrics)} exposure rounds, {n_runs} trigger runs")
    if not series:
        print("Error: No valid RAG series loaded.")
        return
    plot_rag_asr_and_retention(
        series,
        save_dir=save_dir,
        filename_prefix=filename_prefix,
    )
    print("✅ RAG comparison plots generated successfully")


def plot_rag_evolution_comparison(
    series_specs: List[tuple],
    save_dir: str,
    evolution_labels: Optional[List[str]] = None,
    filename: str = "rag_evolution_asr.pdf",
) -> None:
    """
    Load trigger-session metrics for raw / reflection / experience (exactly 3) and plot grouped bar:
    X = Evolution Strategy, Y = ASR, Hue = Attack Task (Data Exfiltration vs Command Execution).
    Uses only trigger session data (no exposure). series_specs: list of 3 (label, trigger_json_path).
    """
    if len(series_specs) != 3:
        print("Error: plot_rag_evolution_comparison requires exactly 3 series (raw, reflection, experience).")
        return
    series = []
    for label, trigger_path in series_specs:
        trigger_metrics, _ = load_metrics_from_json(trigger_path)
        if not trigger_metrics:
            print(f"Warning: No trigger metrics in {trigger_path}, skipping {label}")
            return
        n_runs = len(trigger_metrics)
        series.append((label, [], trigger_metrics))
        print(f"Loaded evolution {label}: {n_runs} trigger runs")
    plot_rag_evolution_asr_grouped(
        series,
        save_dir=save_dir,
        filename=filename,
        evolution_labels=evolution_labels,
    )
    plot_rag_evolution_payload_grouped(
        series,
        save_dir=save_dir,
        filename="rag_evolution_payload_count.pdf",
        evolution_labels=evolution_labels,
    )
    print("✅ RAG evolution ASR and payload count plots generated successfully")


def plot_defense_comparison(
    entries: List[tuple],
    save_dir: str,
    filename: str = "defense_asr_grouped.pdf",
    scenario: str = "RAG",
) -> None:
    """
    Load metrics for each (defense_name, exposure_path, trigger_path) and plot
    grouped bar: X = Instruction guard (raw, raw+sandwich, raw+xxx), Y = ASR, Hue = Data Exfiltration vs Command Execution.
    Works for both RAG and Sliding Window (trigger JSON has same structure: list of runs, each run list of round dicts with exfiltration, command_exec).
    entries: list of (defense_name, exposure_path, trigger_path). defense_name e.g. "raw", "raw+sandwich", "raw+spotlight".
    scenario: "RAG" or "Sliding Window" (for plot title and default filename).
    """
    defense_data = []
    for defense_name, exposure_path, trigger_path in entries:
        trigger_metrics, _ = load_metrics_from_json(trigger_path)
        if not trigger_metrics:
            print(f"Warning: No trigger metrics in {trigger_path}, skipping {defense_name}")
            continue
        exfil_mean, exfil_se, cmd_mean, cmd_se = _exfil_cmd_mean_se_from_trigger(trigger_metrics)
        defense_data.append((
            defense_name,
            [
                ("Data Exfiltration", exfil_mean, exfil_se),
                ("Command Execution", cmd_mean, cmd_se),
            ],
        ))
        print(f"Loaded {defense_name}: Exfil ASR = {exfil_mean:.2%} ± {exfil_se:.2%}, Cmd ASR = {cmd_mean:.2%} ± {cmd_se:.2%}")

    if not defense_data:
        print("Error: No valid defense entries loaded.")
        return
    plot_defense_asr_grouped(defense_data, save_dir=save_dir, filename=filename, scenario=scenario)
    print(f"✅ Instruction defense ASR plot ({scenario}) generated successfully")


def plot_all_from_directory(save_dir: str, model_name: Optional[str] = None) -> None:
    """
    Convenience function to plot all metrics found in a directory.
    Automatically detects whether metrics are RAG or Sliding Window based on file naming.
    
    Args:
        save_dir: Directory containing the metrics JSON files
        model_name: Optional model name for output filename (e.g., "google/gemini-2.5-flash")
    """
    save_dir = Path(save_dir)
    
    if not save_dir.exists():
        print(f"Error: Directory {save_dir} does not exist")
        return
    
    # Generate model nickname for output files
    if model_name:
        model_nick_name = model_name.replace("/", "_")
    else:
        model_nick_name = "model"
    
    # Check for RAG metrics (standard naming)
    exposure_path = save_dir / "exposure_metrics.json"
    trigger_path = save_dir / "trigger_metrics.json"
    
    # Check for Sliding Window metrics (metrics_exposure_*, metrics_trigger_*)
    sw_exposure_files = list(save_dir.glob("metrics_exposure_*.json"))
    sw_trigger_files = list(save_dir.glob("metrics_trigger_*.json"))
    
    # Process RAG metrics if found
    if exposure_path.exists():
        output_path = save_dir / f"exposure_{model_nick_name}.png"
        print("\n" + "="*60)
        print("Processing RAG Exposure Metrics")
        print("="*60)
        plot_from_saved_exposure(str(exposure_path), str(output_path))
    
    if trigger_path.exists():
        output_path = save_dir / f"trigger_{model_nick_name}.png"
        print("\n" + "="*60)
        print("Processing RAG Trigger Metrics")
        print("="*60)
        plot_from_saved_trigger(str(trigger_path), str(output_path))
    
    # Process Sliding Window metrics if found
    if sw_exposure_files and sw_trigger_files:
        print("\n" + "="*60)
        print("Processing Sliding Window Metrics")
        print("="*60)
        
        # Use the first matching pair (or you can process all)
        for sw_exp_path in sw_exposure_files:
            # Extract model name from filename: metrics_exposure_MODEL.json
            exp_model = sw_exp_path.stem.replace("metrics_exposure_", "")
            
            # Find matching trigger file
            sw_trig_path = save_dir / f"metrics_trigger_{exp_model}.json"
            
            if sw_trig_path.exists():
                output_path = save_dir / f"sliding_window_attack_{exp_model}.png"
                print(f"\nProcessing model: {exp_model}")
                plot_from_saved_sliding_window(
                    str(sw_exp_path),
                    str(sw_trig_path),
                    str(output_path)
                )
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    if exposure_path.exists() or trigger_path.exists():
        print("✅ RAG metrics processed")
    if sw_exposure_files and sw_trigger_files:
        print(f"✅ Sliding Window metrics processed ({len(sw_exposure_files)} model(s))")
    if not (exposure_path.exists() or trigger_path.exists() or sw_exposure_files):
        print("⚠️  No metrics files found in directory")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from saved metrics JSON files (RAG or Sliding Window)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot RAG metrics from specific files
  python -m src.analysis.plot_from_saved --metrics_path exposure_metrics.json --output_path exposure.png --plot_type exposure
  python -m src.analysis.plot_from_saved --metrics_path trigger_metrics.json --output_path trigger.png --plot_type trigger
  
  # Plot Sliding Window metrics
  python -m src.analysis.plot_from_saved --exposure_path metrics_exposure_model.json --trigger_path metrics_trigger_model.json --output_path sw.png --plot_type sliding_window
  
  # Plot Sliding Window comparison (ASR + payload retention, ours vs baselines)
  python -m src.analysis.plot_from_saved --sliding_window_series "Ours:path/to/metrics_exposure_ours.json:path/to/metrics_trigger_ours.json" --sliding_window_series "Baseline:path/to/metrics_exposure_base.json:path/to/metrics_trigger_base.json" --sliding_window_save_dir ./plots
  
  # Plot RAG comparison (ASR + recall@k & payload count, ours vs baselines)
  python -m src.analysis.plot_from_saved --rag_series "Ours:path/to/metrics_exposure_ours.json:path/to/metrics_trigger_ours.json" --rag_series "Baseline:path/to/metrics_exposure_base.json:path/to/metrics_trigger_base.json" --rag_save_dir ./plots
  
  # Plot RAG evolution ASR grouped bar (trigger session only; Raw / Reflection / Experience)
  python -m src.analysis.plot_from_saved --evolution_series "Raw History:path/to/raw_trigger.json" --evolution_series "Verbal Reflection:path/to/reflection_trigger.json" --evolution_series "Refined Experience:path/to/experience_trigger.json" --evolution_save_dir ./plots
  
  # Plot instruction defense ASR (RAG and/or Sliding Window; two metrics: Exfil + Cmd per defense)
  python -m src.analysis.plot_from_saved --defense_rag_series "raw:exp.json:trig.json" --defense_rag_series "raw+sandwich:...:..." --defense_save_dir ./plots
  python -m src.analysis.plot_from_saved --defense_sw_series "raw:sw_exp.json:sw_trig.json" --defense_sw_series "raw+sandwich:...:..." --defense_save_dir ./plots
  
  # Plot all metrics in a directory (auto-detects RAG vs Sliding Window)
  python -m src.analysis.plot_from_saved --save_dir /data2/xianglin/zombie_agent/results --model_name google/gemini-2.5-flash
        """
    )
    
    # Option 1: Specify individual files (RAG)
    parser.add_argument("--metrics_path", type=str, 
                       help="Path to the metrics JSON file (for RAG exposure/trigger)")
    parser.add_argument("--output_path", type=str,
                       help="Path to save the output plot")
    parser.add_argument("--plot_type", type=str, choices=["exposure", "trigger", "sliding_window"],
                       help="Type of plot to generate")
    
    # Option 1b: Specify individual files (Sliding Window)
    parser.add_argument("--exposure_path", type=str,
                       help="Path to exposure metrics JSON (for sliding window)")
    parser.add_argument("--trigger_path", type=str,
                       help="Path to trigger metrics JSON (for sliding window)")
    
    # Option 2: Process entire directory
    parser.add_argument("--save_dir", type=str,
                       help="Directory containing metrics JSON files (will plot all found)")
    parser.add_argument("--model_name", type=str,
                       help="Model name for output filename (e.g., google/gemini-2.5-flash)")

    # Option 3: Sliding window comparison (ours + baselines)
    parser.add_argument("--sliding_window_series", type=str, action="append",
                       help="For sliding_window_comparison: 'label:exposure_json_path:trigger_json_path' (repeat for each method)")
    parser.add_argument("--sliding_window_save_dir", type=str,
                       help="Directory to save ASR and retention figures for sliding_window_comparison")

    # Option 4: RAG comparison (ASR + recall & payload count, ours + baselines)
    parser.add_argument("--rag_series", type=str, action="append",
                       help="For RAG comparison: 'label:exposure_json_path:trigger_json_path' (repeat for each method)")
    parser.add_argument("--rag_save_dir", type=str,
                       help="Directory to save RAG ASR and recall/payload figures")

    # Option 5: RAG evolution comparison (grouped bar: Evolution vs ASR, uses trigger session only)
    parser.add_argument("--evolution_series", type=str, action="append",
                       help="Exactly 3: 'label:trigger_path' for raw, reflection, experience (trigger session data)")
    parser.add_argument("--evolution_save_dir", type=str,
                       help="Directory to save RAG evolution ASR grouped bar chart")

    # Option 6: Instruction defense comparison (RAG and/or Sliding Window)
    parser.add_argument("--defense_rag_series", type=str, action="append",
                       help="RAG: 'DefenseName:exposure_path:trigger_path' (e.g. 'raw:exp.json:trig.json')")
    parser.add_argument("--defense_sw_series", type=str, action="append",
                       help="Sliding Window: 'DefenseName:exposure_path:trigger_path' (same format, SW trigger metrics)")
    parser.add_argument("--defense_save_dir", type=str,
                       help="Directory to save defense ASR grouped bar chart(s)")
    
    args = parser.parse_args()
    
    # Parse RAG evolution comparison (exactly 3 series; trigger session only)
    if getattr(args, "evolution_series", None) and args.evolution_save_dir:
        ev_specs = []
        for s in args.evolution_series:
            parts = s.split(":", 1)
            if len(parts) != 2:
                print(f"Invalid --evolution_series (expected label:trigger_path): {s}")
                continue
            ev_specs.append((parts[0].strip(), parts[1].strip()))
        if len(ev_specs) == 3:
            print("\n" + "="*60)
            print("RAG Evolution ASR (Grouped Bar: Raw / Reflection / Experience)")
            print("="*60)
            plot_rag_evolution_comparison(ev_specs, save_dir=args.evolution_save_dir)
            return
        elif ev_specs:
            print("Error: --evolution_series must be given exactly 3 times (raw, reflection, experience).")
            return

    # Parse defense comparison (RAG and/or Sliding Window, two metrics: Exfil + Cmd)
    if args.defense_save_dir:
        if getattr(args, "defense_rag_series", None):
            def_entries = []
            for s in args.defense_rag_series:
                parts = s.split(":", 2)
                if len(parts) != 3:
                    print(f"Invalid --defense_rag_series (expected DefenseName:exposure_path:trigger_path): {s}")
                    continue
                def_entries.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
            if def_entries:
                print("\n" + "="*60)
                print("Instruction Defense ASR (RAG)")
                print("="*60)
                plot_defense_comparison(
                    def_entries,
                    save_dir=args.defense_save_dir,
                    filename="defense_asr_grouped.pdf",
                    scenario="RAG",
                )
                if not getattr(args, "defense_sw_series", None):
                    return
        if getattr(args, "defense_sw_series", None):
            def_sw_entries = []
            for s in args.defense_sw_series:
                parts = s.split(":", 2)
                if len(parts) != 3:
                    print(f"Invalid --defense_sw_series (expected DefenseName:exposure_path:trigger_path): {s}")
                    continue
                def_sw_entries.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
            if def_sw_entries:
                print("\n" + "="*60)
                print("Instruction Defense ASR (Sliding Window)")
                print("="*60)
                plot_defense_comparison(
                    def_sw_entries,
                    save_dir=args.defense_save_dir,
                    filename="defense_sw_asr_grouped.pdf",
                    scenario="Sliding Window",
                )
            return
        if getattr(args, "defense_rag_series", None) or getattr(args, "defense_sw_series", None):
            return
    
    # Parse RAG comparison
    if getattr(args, "rag_series", None) and args.rag_save_dir:
        series_specs = []
        for s in args.rag_series:
            parts = s.split(":", 2)
            if len(parts) != 3:
                print(f"Invalid --rag_series (expected label:exposure_path:trigger_path): {s}")
                continue
            series_specs.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
        if series_specs:
            print("\n" + "="*60)
            print("RAG Comparison (ASR + Recall & Payload Count)")
            print("="*60)
            plot_rag_comparison(series_specs, save_dir=args.rag_save_dir)
            return
    
    # Parse sliding_window_comparison
    if getattr(args, "sliding_window_series", None) and args.sliding_window_save_dir:
        series_specs = []
        for s in args.sliding_window_series:
            parts = s.split(":", 2)
            if len(parts) != 3:
                print(f"Invalid --sliding_window_series (expected label:exposure_path:trigger_path): {s}")
                continue
            series_specs.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
        if series_specs:
            print("\n" + "="*60)
            print("Sliding Window Comparison (ASR + Payload Retention)")
            print("="*60)
            plot_sliding_window_comparison(
                series_specs,
                save_dir=args.sliding_window_save_dir,
            )
            return
    
    # Validate arguments
    if args.save_dir:
        # Option 2: Process directory
        plot_all_from_directory(args.save_dir, args.model_name)
    elif args.plot_type == "sliding_window" and args.exposure_path and args.trigger_path and args.output_path:
        # Option 1b: Sliding Window
        print("\n" + "="*60)
        print("Processing Sliding Window Metrics")
        print("="*60)
        plot_from_saved_sliding_window(args.exposure_path, args.trigger_path, args.output_path)
    elif args.metrics_path and args.output_path and args.plot_type in ["exposure", "trigger"]:
        # Option 1a: RAG
        print("\n" + "="*60)
        print(f"Processing RAG {args.plot_type.capitalize()} Metrics")
        print("="*60)
        
        if args.plot_type == "exposure":
            plot_from_saved_exposure(args.metrics_path, args.output_path)
        else:  # trigger
            plot_from_saved_trigger(args.metrics_path, args.output_path)
    else:
        parser.print_help()
        print("\n❌ Error: Invalid argument combination. Use one of:")
        print("  1. --save_dir [--model_name]")
        print("  2. --metrics_path --output_path --plot_type [exposure|trigger]")
        print("  3. --exposure_path --trigger_path --output_path --plot_type sliding_window")
        print("  4. --sliding_window_series label:exposure_path:trigger_path (repeat) --sliding_window_save_dir DIR")
        print("  5. --rag_series label:exposure_path:trigger_path (repeat) --rag_save_dir DIR")
        print("  6. --evolution_series label:trigger_path (exactly 3, trigger session) --evolution_save_dir DIR")
        print("  7. --defense_rag_series / --defense_sw_series DefenseName:exposure_path:trigger_path (repeat) --defense_save_dir DIR")


if __name__ == "__main__":
    main()

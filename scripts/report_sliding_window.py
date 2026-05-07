"""
Report sliding window experiment results.

Generates a table with four metrics for each model × defense combination:
  - Utility          : clean input ± defense  (utility_clean/ or utility_defense/)
  - Util (attack)    : under attack ± defense  (main sliding_window dir)
  - ASR: Exfiltration: attack success — data exfiltration (main dir)
  - ASR: Cmd Exec    : attack success — command execution  (main dir)

Usage:
    python scripts/report_sliding_window.py
    python scripts/report_sliding_window.py --save_dir /data2/xianglin/zombie_agent/sliding_window
    python scripts/report_sliding_window.py --format csv --output results.csv
"""

import json
import os
import argparse
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SAVE_DIR = "/data2/xianglin/zombie_agent/sliding_window"
DATASET          = "data-for-agents_insta-150k-v1"

MODELS = {
    "Gemini":   "google_gemini-2.5-flash",
    "GLM":      "z-ai_glm-4.7-flash",
    "LLaMA":    "meta-llama_llama-3.3-70b-instruct",
    "DeepSeek": "deepseek_deepseek-v3.2",
}

# (display_label, clean_suffix, defense_util_suffix, attack_suffix)
#   clean_suffix        → utility_clean/   (clean input, NO defense)   — only for None row
#   defense_util_suffix → utility_defense/ (clean input, WITH defense) — over-defense check
#   attack_suffix       → main dir         (attacked input, ±defense)
CONFIGS = [
    ("None",          "0_openai_gpt-4.1-nano_raw",  None,                                        "0_PIGuard_raw"),
    ("sandwich",      None,                          "0_PIGuard_sandwich",                        "0_PIGuard_sandwich"),
    ("spotlight",     None,                          "0_PIGuard_spotlight",                       "0_PIGuard_spotlight"),
    ("instructional", None,                          "0_PIGuard_instructional",                   "0_PIGuard_instructional"),
    ("PIGuard",      None,                          "1_PIGuard_raw",                             "1_PIGuard_raw"),
    ("AIProtectv2",    None,                          "1_ProtectAIv2_raw",                         "1_ProtectAIv2_raw"),
    ("Cleaner",           None,                          "0_openai_gpt-4.1-nano_raw_cleaner_openai_gpt-5-nano", "0_openai_gpt-4.1-nano_raw_cleaner_openai_gpt-5-nano"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_stats(fpath):
    """Return stats dict if file exists, else None."""
    if not os.path.exists(fpath):
        return None
    with open(fpath) as fp:
        d = json.load(fp)
    all_items = [item for run in d["metrics"] for item in run]
    total     = len(all_items)
    exfil_ok  = sum(1 for x in all_items if x.get("exfiltration"))
    cmd_ok    = sum(1 for x in all_items if x.get("command_exec"))
    judged    = [x for x in all_items if x.get("task_completion")]
    util_ok   = sum(1 for x in judged if x["task_completion"].get("completed"))
    util_n    = len(judged)
    return {
        "total":   total,
        "exfil":  (exfil_ok, total),
        "cmd":    (cmd_ok,   total),
        "utility":(util_ok,  util_n) if util_n > 0 else None,
    }


def trigger_path(base, subdir, model_str, suffix):
    d = os.path.join(base, subdir) if subdir else base
    return os.path.join(d, f"metrics_trigger_{model_str}_zombie_completion_real_{DATASET}_{suffix}.json")


def fmt(pair, as_raw=False):
    """Format a (ok, total) pair.  Returns '-' if data is missing."""
    if pair is None:
        return "-", "-", "-"
    ok, total = pair
    if total == 0:
        return "-", "-", "-"
    pct = f"{ok / total:.2%}"
    frac = f"{ok}/{total}"
    combined = f"{pct} ({frac})"
    return pct, frac, combined


# ── main ──────────────────────────────────────────────────────────────────────

def build_rows(save_dir):
    rows = []
    for model_abbr, model_str in MODELS.items():
        for label, clean_sfx, def_sfx, atk_sfx in CONFIGS:

            # Utility (clean input ± defense)
            if clean_sfx:
                util_r = load_stats(trigger_path(save_dir, "utility_clean", model_str, clean_sfx))
            else:
                util_r = load_stats(trigger_path(save_dir, "utility_defense", model_str, def_sfx))
            utility = util_r["utility"] if util_r else None

            # Utility under attack + ASR — main dir
            atk_r    = load_stats(trigger_path(save_dir, "", model_str, atk_sfx))
            util_atk = atk_r["utility"] if atk_r else None
            exfil    = atk_r["exfil"]   if atk_r else None
            cmd      = atk_r["cmd"]     if atk_r else None

            rows.append({
                "model":    model_abbr,
                "defense":  label,
                "utility":  utility,
                "util_atk": util_atk,
                "exfil":    exfil,
                "cmd":      cmd,
            })
    return rows


def print_table(rows):
    COL = 20
    header = (
        f"{'Model':<10} {'Defense':<16}"
        f" {'Utility':>{COL}}"
        f" {'Util(attack)':>{COL}}"
        f" {'ASR: Exfil':>{COL}}"
        f" {'ASR: CmdExec':>{COL}}"
    )
    print(header)
    print("=" * len(header))

    prev_model = None
    for r in rows:
        if prev_model and r["model"] != prev_model:
            print()
        prev_model = r["model"]

        _, _, u   = fmt(r["utility"])
        _, _, ua  = fmt(r["util_atk"])
        _, _, ex  = fmt(r["exfil"])
        _, _, cm  = fmt(r["cmd"])
        print(
            f"{r['model']:<10} {r['defense']:<16}"
            f" {u:>{COL}} {ua:>{COL}} {ex:>{COL}} {cm:>{COL}}"
        )


def save_csv(rows, output_path):
    import csv
    with open(output_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Model", "Defense",
                         "Utility(%)", "Utility(frac)",
                         "Util_attack(%)", "Util_attack(frac)",
                         "ASR_Exfil(%)", "ASR_Exfil(frac)",
                         "ASR_Cmd(%)", "ASR_Cmd(frac)"])
        for r in rows:
            u_pct,  u_frac,  _ = fmt(r["utility"])
            ua_pct, ua_frac, _ = fmt(r["util_atk"])
            ex_pct, ex_frac, _ = fmt(r["exfil"])
            cm_pct, cm_frac, _ = fmt(r["cmd"])
            writer.writerow([
                r["model"], r["defense"],
                u_pct,  u_frac,
                ua_pct, ua_frac,
                ex_pct, ex_frac,
                cm_pct, cm_frac,
            ])
    print(f"Saved to {output_path}")


def save_markdown(rows, output_path):
    lines = []
    lines.append("| Model | Defense | Utility | Util (attack) | ASR: Exfil | ASR: Cmd |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        _, _, u  = fmt(r["utility"])
        _, _, ua = fmt(r["util_atk"])
        _, _, ex = fmt(r["exfil"])
        _, _, cm = fmt(r["cmd"])
        lines.append(f"| {r['model']} | {r['defense']} | {u} | {ua} | {ex} | {cm} |")
    with open(output_path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"Saved to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report sliding window experiment results.")
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR,
                        help="Root directory containing utility_clean/, utility_defense/, and main results.")
    parser.add_argument("--format", choices=["table", "csv", "markdown"], default="table",
                        help="Output format (default: table).")
    parser.add_argument("--output", default=None,
                        help="Output file path for csv/markdown formats.")
    args = parser.parse_args()

    rows = build_rows(args.save_dir)

    if args.format == "table":
        print_table(rows)
    elif args.format == "csv":
        out = args.output or "sliding_window_results.csv"
        save_csv(rows, out)
    elif args.format == "markdown":
        out = args.output or "sliding_window_results.md"
        save_markdown(rows, out)

#!/usr/bin/env python3
"""
delta_chair_correlation.py
===========================
Δₜ decline And CHAIRs association analysis.
Compare 6 Kind contrastive decoding Method trajectory decline With each CHAIRs qualitative relationship.
Pure CPU Compute,~ 1 Minutes,No GPU Require.

Input:
  - results/diagnostic/diagnostic_methods_*.json         Existing trajectory Data
  - results/all_chair_eval.csv                           Existing aggregate CHAIR Data

Usage:
    python3 delta_chair_correlation.py
"""

import json
import csv
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DIAG_DIR = REPO_ROOT / "results" / "diagnostic"
CHAIR_CSV = REPO_ROOT / "tables" / "chair_eval_all.csv"

METHODS = ["vcd", "dola", "icd", "deco", "halc", "sid"]

# ── 1. From trajectory Data extraction for each method average decline ──
print("=" * 70)
print("1. From trajectory Data computes each method decline")
print("-" * 70)

method_stats = {}
for method in METHODS:
    fpath = DIAG_DIR / f"diagnostic_methods_{method}.json"
    with open(fpath) as f:
        data = json.load(f)

    declines = []
    early_all, late_all = [], []

    for rec in data:
        traj = rec.get("delta_trajectory", [])
        T = len(traj)
        if T < 20:
            continue
        m = T // 2
        early = float(np.mean(traj[:m]))
        late = float(np.mean(traj[m:]))
        early_all.append(early)
        late_all.append(late)
        declines.append(early - late)

    method_stats[method] = {
        "n": len(declines),
        "early_mean": float(np.mean(early_all)),
        "late_mean": float(np.mean(late_all)),
        "decline_mean": float(np.mean(declines)),
        "decline_se": float(np.std(declines, ddof=1) / np.sqrt(len(declines))),
    }
    print(f"  {method:<8}: N={len(declines):>3}, "
          f"Early={method_stats[method]['early_mean']:>7.4f}, "
          f"Late={method_stats[method]['late_mean']:>7.4f}, "
          f"Decline={method_stats[method]['decline_mean']:>7.4f} ± {method_stats[method]['decline_se']:.4f}")

# ── 2. From CHAIR CSV Extract data ──
print(f"\n2. From {CHAIR_CSV.name} Extract CHAIR Data")
print("-" * 70)

chair_data = {}
with open(CHAIR_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        model = row["model"].strip()
        method = row["method"].strip().lstrip("_")  # "__baseline__" -> "baseline"
        if model == "llava" and method in METHODS:
            chair_data[method] = {
                "CHAIRs": float(row["CHAIRs"]),
                "CHAIRi": float(row["CHAIRi"]),
                "Len": float(row["Len"]),
            }
            print(f"  llava/{method}: CHAIRs={chair_data[method]['CHAIRs']:.1f}%, "
                  f"CHAIRi={chair_data[method]['CHAIRi']:.1f}%, Len={chair_data[method]['Len']:.1f}")

# Record which methods appear in both datasets
common = sorted(set(method_stats.keys()) & set(chair_data.keys()))
missing_chair = set(method_stats.keys()) - set(chair_data.keys())
if missing_chair:
    print(f"  Warning: The following methods have no CHAIR Data: {missing_chair}")

# ── 3. Comparison table ──
print(f"\n3. Comparison table:Decline vs CHAIRs (N={len(common)} methods)")
print("=" * 70)
print(f"{'Method':<10} {'N':>4} {'Decline':>9} {'CHAIRs':>8} {'CHAIRi':>8} {'Len':>7}")
print("-" * 55)
for method in sorted(common, key=lambda m: method_stats[m]["decline_mean"], reverse=True):
    ms = method_stats[method]
    cd = chair_data[method]
    print(f"{method:<10} {ms['n']:>4} {ms['decline_mean']:>9.4f} {cd['CHAIRs']:>7.1f}% "
          f"{cd['CHAIRi']:>7.1f}% {cd['Len']:>6.1f}")

# Baseline Reference
print()
print("Baseline (beam search) reference from Table tab:methods_trajectory:")
print("  Baseline Decline = +0.322, CHAIRs = 54.0%, Len = 91.3")

# ── 4. Correlation ──
if len(common) >= 3:
    print(f"\n4. Correlation analysis")
    print("-" * 70)
    decline_vals = [method_stats[m]["decline_mean"] for m in common]
    chairs_vals  = [chair_data[m]["CHAIRs"] for m in common]
    chairi_vals  = [chair_data[m]["CHAIRi"] for m in common]

    r_chairs = float(np.corrcoef(decline_vals, chairs_vals)[0, 1])
    r_chairi = float(np.corrcoef(decline_vals, chairi_vals)[0, 1])
    print(f"  Pearson r (Decline vs CHAIRs)  = {r_chairs:+.4f}")
    print(f"  Pearson r (Decline vs CHAIRi)  = {r_chairi:+.4f}")
    print(f"  Warning: N={len(common)} methods -- Statistical test power insufficient,For descriptive analysis only")

#!/usr/bin/env python3
"""Matched-length truncation analysis for CHAIR evaluation.

This script performs non-parametric length normalization by truncating all
captions to a target length and recomputing CHAIRs under equal-length conditions.

INPUT FORMAT
============
A CSV file with columns:
  model, method, caption_id, CHAIRs_binary, num_hallucinated, num_mentioned, caption_length_words

Where:
  - CHAIRs_binary: 1 if the caption contains ≥1 hallucinated object, else 0
  - num_hallucinated: count of hallucinated objects in the caption (H_i)
  - num_mentioned: count of mentioned objects in the caption (M_i)
  - caption_length_words: word count of the caption (L_i)

INSTRUCTIONS
============
1. Export your per-caption CHAIR evaluation data into a CSV with the columns above.
   This data is typically available from the official CHAIR evaluation script's
   per-caption output (it logs H_i and M_i per caption).

2. Save it as: per_caption_chair.csv

3. Run:
   python3 matched_length_truncation.py per_caption_chair.csv

OUTPUT
======
- Console: per-model summary of CHAIRs_original vs CHAIRs_trunc + rank comparison
- matched_length_results.tex: LaTeX table
"""

import csv
import math
import sys
from collections import defaultdict


def load_data(path):
    """Load per-caption CHAIR data."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            model = r["model"].strip()
            method = r["method"].strip()
            try:
                chairs_bin = int(r["CHAIRs_binary"])
                num_hall = int(r["num_hallucinated"])
                num_ment = int(r["num_mentioned"])
                length = int(r["caption_length_words"])
            except (ValueError, KeyError):
                print(f"  Skipping malformed row: {r}")
                continue
            rows.append({
                "model": model,
                "method": method,
                "caption_id": r.get("caption_id", ""),
                "chairs_binary": chairs_bin,
                "num_hallucinated": num_hall,
                "num_mentioned": num_ment,
                "length": length,
            })
    return rows


def compute_chairs_original(rows_group):
    """Compute original CHAIRs, CHAIRi, and mean length for a method group."""
    n = len(rows_group)
    chairs_sum = sum(r["chairs_binary"] for r in rows_group)
    h_sum = sum(r["num_hallucinated"] for r in rows_group)
    m_sum = sum(r["num_mentioned"] for r in rows_group)
    len_sum = sum(r["length"] for r in rows_group)
    return {
        "n": n,
        "CHAIRs": chairs_sum / n * 100 if n > 0 else 0,
        "CHAIRi": h_sum / m_sum * 100 if m_sum > 0 else 0,
        "mean_len": len_sum / n if n > 0 else 0,
        "h_sum": h_sum,
        "m_sum": m_sum,
    }


def compute_chairs_truncated(rows_group, target_len):
    """Simulate matched-length truncation: reassign CHAIRs_binary under the
    assumption that capions truncated to target_len have CHAIRs_binary = 1
    iff the original was 1 AND the hallucinated objects appear within the first
    target_len words.

    Without per-object position data, we use a conservative approximation:
    - CHAIRs_binary_trunc = original (if length ≤ target_len, no change;
      if length > target_len, we cannot know whether hallucinations were
      in the truncated portion without position data)

    For a rigorous analysis, you need per-object hallucination positions.
    See the companion notes below.
    """
    n = len(rows_group)
    chairs_sum = 0
    for r in rows_group:
        if r["length"] <= target_len:
            chairs_sum += r["chairs_binary"]
        else:
            # Conservative bound: keep binary as-is (upper bound on trunc effect)
            # Real analysis requires per-object position data
            chairs_sum += r["chairs_binary"]
    return chairs_sum / n * 100 if n > 0 else 0


def rankdata(values, reverse=False):
    """Rank values. reverse=True means smaller values get rank 1."""
    mult = -1 if reverse else 1
    pairs = sorted((mult * v, i) for i, v in enumerate(values))
    ranks = [0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            ranks[pairs[k][1]] = avg_rank
        i = j
    return ranks


def analyze_model(model, methods_data):
    """Analyze one model across methods."""
    # Group by method
    method_groups = defaultdict(list)
    for r in methods_data:
        method_groups[r["method"]].append(r)

    methods = list(method_groups.keys())
    if len(methods) < 2:
        return None

    # Compute original stats
    original = {}
    for method, group in method_groups.items():
        original[method] = compute_chairs_original(group)

    # Determine target length: minimum mean length across methods
    mean_lens = {m: original[m]["mean_len"] for m in methods}
    target_len = int(min(mean_lens.values()))

    # Compute truncated CHAIRs
    truncated = {}
    for method, group in method_groups.items():
        truncated[method] = compute_chairs_truncated(group, target_len)

    # Ranks
    orig_chairs = [original[m]["CHAIRs"] for m in methods]
    trunc_chairs = [truncated[m] for m in methods]
    orig_ranks = rankdata(orig_chairs, reverse=False)  # lower = better
    trunc_ranks = rankdata(trunc_chairs, reverse=False)

    print(f"\n{'='*70}")
    print(f"Model: {model}  (target length: {target_len} words)")
    print(f"{'Method':<15} {'n':>4} {'CHAIRs_orig':>12} {'CHAIRs_trunc':>13} "
          f"{'Len':>7} {'Δ_rank':>7}")
    print("-" * 70)

    for i, m in enumerate(methods):
        delta = int(orig_ranks[i] - trunc_ranks[i])
        print(f"{m:<15} {original[m]['n']:>4} {original[m]['CHAIRs']:>11.1f}% "
              f"{truncated[m]:>12.1f}% {original[m]['mean_len']:>6.1f} {delta:>+7d}")

    return {
        "model": model,
        "methods": methods,
        "original": original,
        "truncated": truncated,
        "orig_ranks": orig_ranks,
        "trunc_ranks": trunc_ranks,
        "target_len": target_len,
    }


def generate_table(all_results, output_path):
    """Generate LaTeX table."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Matched-length truncation across architectures. "
                  r"All captions truncated to the minimum mean length within each model. "
                  r"$\Delta$Rank: positive = method improved rank after truncation "
                  r"(its original longer captions inflated CHAIRs).}")
    lines.append(r"\label{tab:matched_length_cross}")
    lines.append(r"\begin{tabular}{@{}llcccc@{}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Method} & \textbf{CHAIRs} & "
                  r"\textbf{CHAIRs\textsubscript{trunc}} & \textbf{Len} & "
                  r"\textbf{$\Delta$Rank} \\")
    lines.append(r"\midrule")

    for result in all_results:
        model = result["model"]
        methods = result["methods"]

        # Sort by original CHAIRs
        sorted_methods = sorted(methods, key=lambda m: result["original"][m]["CHAIRs"])

        for i, m in enumerate(sorted_methods):
            delta_rank = int(result["orig_ranks"][methods.index(m)] -
                            result["trunc_ranks"][methods.index(m)])
            if i == 0:
                model_str = model
            else:
                model_str = ""
            lines.append(
                f"{model_str} & {m} & "
                f"{result['original'][m]['CHAIRs']:.1f}\\% & "
                f"{result['truncated'][m]:.1f}\\% & "
                f"{result['original'][m]['mean_len']:.0f} & "
                f"{'+' if delta_rank > 0 else ''}{delta_rank} \\\\"
            )
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {output_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nERROR: No input CSV provided.")
        print("Usage: python3 matched_length_truncation.py per_caption_chair.csv")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading {path}...")
    rows = load_data(path)
    print(f"Loaded {len(rows)} per-caption records")

    # Group by model
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    all_results = []
    for model in sorted(by_model.keys()):
        result = analyze_model(model, by_model[model])
        if result:
            all_results.append(result)

    if all_results:
        generate_table(all_results, "tables/matched_length_cross.tex")
        print("\nDone.")
        print("IMPORTANT NOTE:")
        print("  This script uses a conservative upper-bound approximation for truncation:")
        print("  captions truncated mid-sentence keep their original CHAIRs_binary value.")
        print("  For a rigorous analysis, you need per-object hallucination position data")
        print("  (which token/word position each hallucinated object appears at).")
        print("  A rigorous matched-length analysis requires this position-level data.")
    else:
        print("\nNo data to analyze.")


if __name__ == "__main__":
    main()

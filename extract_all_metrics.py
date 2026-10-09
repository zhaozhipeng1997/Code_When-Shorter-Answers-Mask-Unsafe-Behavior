#!/usr/bin/env python3
"""
Extract ALL baseline metrics from results/baseline/
and aggregate into a structured summary JSON.

File naming patterns (regex-based):
  hallusion_{model}_{method}_result.json   -- dict with qAcc, fAcc, aAcc, consistency, etc.
  hallusion_{model}_{method}_eval_v{s|d}.json  -- eval files with 'same' and 'gpt4v_output_gpt_check'
  mmhal_{model}_{method}_result.json       -- dict with "Average score", "Hallucination rate", per-category
  chair_{model}_{method}.jsonl             -- JSON lines with image_id, caption
  pope_{subset}_{model}_{method}.json      -- list of dicts with question_id, image, question, response

Output: results/baseline_summary.json
"""

import json
import re
import os
from collections import defaultdict

BASELINE_DIR = os.path.join(os.path.dirname(__file__), "results/baseline")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "results/baseline_summary.json")

# Regex patterns to parse filenames
RE_HALLUSION_RESULT = re.compile(r'^hallusion_(.+)_(.+)_result\.json$')
RE_HALLUSION_EVAL = re.compile(r'^hallusion_(.+)_(.+)_eval_v([sd])\.json$')
RE_MMHAL_RESULT = re.compile(r'^mmhal_(.+)_(.+)_result\.json$')
RE_CHAIR = re.compile(r'^chair_(.+)_(.+)\.jsonl$')
RE_POPE = re.compile(r'^pope_(.+)_(.+)_(.+)\.json$')

def scan_files():
    """Scan the baseline directory and group files by type and (model, method)."""
    files = os.listdir(BASELINE_DIR)

    hallusion_results = {}   # (model, method) -> path
    hallusion_evals = {}     # (model, method, variant) -> path
    mmhal_results = {}       # (model, method) -> path
    chair_files = {}         # (model, method) -> path
    pope_files = {}          # (subset, model, method) -> path

    for fname in files:
        fpath = os.path.join(BASELINE_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        m = RE_HALLUSION_RESULT.match(fname)
        if m:
            model, method = m.group(1), m.group(2)
            hallusion_results[(model, method)] = fpath
            continue

        m = RE_HALLUSION_EVAL.match(fname)
        if m:
            model, method, variant = m.group(1), m.group(2), m.group(3)
            hallusion_evals[(model, method, variant)] = fpath
            continue

        m = RE_MMHAL_RESULT.match(fname)
        if m:
            model, method = m.group(1), m.group(2)
            mmhal_results[(model, method)] = fpath
            continue

        m = RE_CHAIR.match(fname)
        if m:
            model, method = m.group(1), m.group(2)
            chair_files[(model, method)] = fpath
            continue

        m = RE_POPE.match(fname)
        if m:
            subset, model, method = m.group(1), m.group(2), m.group(3)
            pope_files[(subset, model, method)] = fpath
            continue

    return hallusion_results, hallusion_evals, mmhal_results, chair_files, pope_files

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_jsonl(path):
    """Load a JSONL file, returning list of dicts."""
    items = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def extract_hallusion_result(path):
    """Extract metrics from a hallusion_{model}_{method}_result.json."""
    data = load_json(path)
    return {
        "qAcc": data.get("qAcc"),
        "fAcc": data.get("fAcc"),
        "aAcc": data.get("aAcc"),
        "easy_aAcc": data.get("easy_aAcc"),
        "hard_aAcc": data.get("hard_aAcc"),
        "YesNo_diff": data.get("YesNo_diff"),
        "YesNo_fp_ratio": data.get("YesNo_fp_ratio"),
        "consistency_correct": data.get("consistency_correct"),
        "consistency_inconsistent": data.get("consistency_inconsistent"),
        "consistency_wrong": data.get("consistency_wrong"),
        "LH_ratio": data.get("LH_ratio"),
        "VI_ratio": data.get("VI_ratio"),
        "Mixed_ratio": data.get("Mixed_ratio"),
        "wrong_total": data.get("wrong_total"),
    }

def extract_hallusion_eval(path):
    """Extract stats from a hallusion eval file.
    Returns count of items, count where same=True, count where same=False."""
    data = load_json(path)
    total = len(data)
    same_count = sum(1 for item in data if item.get("same") is True)
    not_same_count = total - same_count
    return {
        "total": total,
        "same": same_count,
        "not_same": not_same_count,
    }

def extract_mmhal_result(path):
    """Extract metrics from mmhal_{model}_{method}_result.json."""
    data = load_json(path)
    # Known category keys
    categories = [
        "Object Attribute", "Adversarial Object", "Comparison",
        "Counting", "Spatial Relation", "Environment",
        "Holistic Description", "Other"
    ]
    result = {
        "Average score": data.get("Average score"),
        "Hallucination rate": data.get("Hallucination rate"),
        "valid_ratings": data.get("valid_ratings"),
        "total": data.get("total"),
    }
    for cat in categories:
        if cat in data:
            result[cat] = data[cat]
    return result

def extract_chair_stats(path):
    """Extract caption statistics from a .jsonl file."""
    items = load_jsonl(path)
    num_captions = len(items)
    if num_captions == 0:
        return {"num_captions": 0, "mean_caption_length": 0, "total_words": 0}
    total_words = sum(len(item.get("caption", "").split()) for item in items)
    mean_length = total_words / num_captions
    return {
        "num_captions": num_captions,
        "mean_caption_length": round(mean_length, 2),
        "total_words": total_words,
    }

def extract_pope_stats(path):
    """Extract response statistics from a pope file.
    Counts total responses, yes/no ratio from the 'response' field."""
    data = load_json(path)
    total = len(data)
    yes_count = 0
    no_count = 0
    for item in data:
        resp = item.get("response", "")
        resp_lower = resp.lower().strip()

        detected = False

        # Strategy 1: after the last '?' in the response
        if '?' in resp_lower:
            after_question = resp_lower.split('?', 1)[1].strip()
            if after_question.startswith("yes"):
                yes_count += 1
                detected = True
            elif after_question.startswith("no"):
                no_count += 1
                detected = True

        # Strategy 2: starts with yes/no directly (no question mark)
        if not detected:
            if resp_lower.startswith("yes"):
                yes_count += 1
                detected = True
            elif resp_lower.startswith("no"):
                no_count += 1
                detected = True

        # Strategy 3: "the answer is yes/no"
        if not detected:
            if re.search(r'\banswer is yes\b', resp_lower):
                yes_count += 1
                detected = True
            elif re.search(r'\banswer is no\b', resp_lower):
                no_count += 1
                detected = True

        # Strategy 4: quoted yes/no
        if not detected:
            if '"yes"' in resp_lower or "'yes'" in resp_lower:
                yes_count += 1
                detected = True
            elif '"no"' in resp_lower or "'no'" in resp_lower:
                no_count += 1
                detected = True

        # Strategy 5: implicit negation patterns
        if not detected:
            if re.search(r'\bcannot (confidently )?(confirm|determine|see)\b', resp_lower):
                no_count += 1
                detected = True
            elif re.search(r'\bno mention of\b', resp_lower):
                no_count += 1
                detected = True
            elif re.search(r'\bno visible\b', resp_lower):
                no_count += 1
                detected = True
            elif re.search(r'\bnot visible\b', resp_lower):
                no_count += 1
                detected = True
            elif re.search(r'\bnot discernible\b', resp_lower):
                no_count += 1
                detected = True
            elif re.search(r'\bthere is no\b', resp_lower):
                no_count += 1
                detected = True
            elif re.search(r'\bthere are no\b', resp_lower):
                no_count += 1
                detected = True
            elif re.search(r'\bdoes not (appear|contain|include|show|depict|seem|have)\b', resp_lower):
                no_count += 1
                detected = True
            elif re.search(r'\bnot possible to\b', resp_lower):
                no_count += 1
                detected = True

        # Strategy 6: Chinese answer patterns in response / qwen-style
        if not detected:
            # Chinese: Answer is affirmative (yes) / Answer is negative (no) / No (no) / Yes (yes)
            if re.search(r'Answer is affirmative|Answer is.*Is|.*Has.*Exists', resp_lower):
                yes_count += 1
                detected = True
            elif re.search(r'Answer is negative|Answer is.*No|Answer.*Negation|No.*In|Not found|Not.*Exists', resp_lower):
                no_count += 1
                detected = True

        # Strategy 7: Check answer field
        if not detected:
            ans = str(item.get("answer", "")).strip().lower()
            if ans == "yes":
                yes_count += 1
            elif ans == "no":
                no_count += 1

    yes_ratio = yes_count / total if total > 0 else 0
    no_ratio = no_count / total if total > 0 else 0

    return {
        "total_responses": total,
        "yes_count": yes_count,
        "no_count": no_count,
        "other_count": total - yes_count - no_count,
        "yes_ratio": round(yes_ratio, 4),
        "no_ratio": round(no_ratio, 4),
    }

def main():
    print(f"Scanning {BASELINE_DIR} ...")
    hallusion_results, hallusion_evals, mmhal_results, chair_files, pope_files = scan_files()

    print(f"  Hallusion result files: {len(hallusion_results)}")
    print(f"  Hallusion eval files:   {len(hallusion_evals)}")
    print(f"  MMHal result files:     {len(mmhal_results)}")
    print(f"  CHAIR files:            {len(chair_files)}")
    print(f"  POPE files:             {len(pope_files)}")

    # Build summary structure
    summary = {
        "hallusion": {},
        "mmhal": {},
        "chair": {},
        "pope": {},
    }

    # Process Hallusion results
    print("\nProcessing Hallusion results...")
    for (model, method), path in sorted(hallusion_results.items()):
        metrics = extract_hallusion_result(path)
        key = f"{model}__{method}"
        if model not in summary["hallusion"]:
            summary["hallusion"][model] = {}
        summary["hallusion"][model][method] = metrics

    # Process Hallusion evals
    print("Processing Hallusion evals...")
    hallusion_eval_data = {}
    for (model, method, variant), path in sorted(hallusion_evals.items()):
        stats = extract_hallusion_eval(path)
        key = f"{model}__{method}__v{variant}"
        hallusion_eval_data[key] = stats

    # Process MMHal results
    print("Processing MMHal results...")
    for (model, method), path in sorted(mmhal_results.items()):
        metrics = extract_mmhal_result(path)
        if model not in summary["mmhal"]:
            summary["mmhal"][model] = {}
        summary["mmhal"][model][method] = metrics

    # Process CHAIR files
    print("Processing CHAIR files...")
    for (model, method), path in sorted(chair_files.items()):
        stats = extract_chair_stats(path)
        if model not in summary["chair"]:
            summary["chair"][model] = {}
        summary["chair"][model][method] = stats

    # Process POPE files
    print("Processing POPE files...")
    for (subset, model, method), path in sorted(pope_files.items()):
        stats = extract_pope_stats(path)
        if subset not in summary["pope"]:
            summary["pope"][subset] = {}
        if model not in summary["pope"][subset]:
            summary["pope"][subset][model] = {}
        summary["pope"][subset][model][method] = stats

    # Add eval data to hallusion section
    summary["hallusion_evals"] = hallusion_eval_data

    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary["_stats"] = {
        "num_hallusion_result_files": len(hallusion_results),
        "num_hallusion_eval_files": len(hallusion_evals),
        "num_mmhal_result_files": len(mmhal_results),
        "num_chair_files": len(chair_files),
        "num_pope_files": len(pope_files),
        "models_found": sorted(set(
            [m for m, _ in hallusion_results] +
            [m for m, _ in mmhal_results] +
            [m for m, _ in chair_files] +
            [m for _, m, _ in pope_files]
        )),
        "methods_found": sorted(set(
            [mth for _, mth in hallusion_results] +
            [mth for _, mth in mmhal_results] +
            [mth for _, mth in chair_files] +
            [mth for _, _, mth in pope_files]
        )),
        "pope_subsets": sorted(set(s for s, _, _ in pope_files)),
    }

    # Write output
    print(f"\nWriting summary to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(summary, f, indent=2)

    print("Done!")
    print(f"\nSummary structure:")
    print(f"  hallusion: {len(summary['hallusion'])} models, "
          f"{sum(len(v) for v in summary['hallusion'].values())} entries")
    print(f"  hallusion_evals: {len(summary['hallusion_evals'])} entries")
    print(f"  mmhal: {len(summary['mmhal'])} models, "
          f"{sum(len(v) for v in summary['mmhal'].values())} entries")
    print(f"  chair: {len(summary['chair'])} models, "
          f"{sum(len(v) for v in summary['chair'].values())} entries")
    for subset in summary["pope"]:
        entries = sum(len(v) for v in summary["pope"][subset].values())
        print(f"  pope/{subset}: {len(summary['pope'][subset])} models, {entries} entries")
    print(f"\nModels: {summary['_stats']['models_found']}")
    print(f"Methods: {summary['_stats']['methods_found']}")

if __name__ == "__main__":
    main()

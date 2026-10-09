#!/usr/bin/env python3
"""
=============================================================
 Per-Caption CHAIR Evaluation Pipeline
=============================================================

Computes per-caption hallucination statistics (H_i, M_i, L_i)
from model-generated captions against COCO ground-truth annotations.

INPUT
-----
A directory of JSONL files, one per model×method combination, named like:
    llava_vcd.jsonl
    llava_opera.jsonl
    qwen_baseline.jsonl
    ...

Each JSONL file has one JSON object per line:
    {"image_id": 139, "caption": "A cat sitting on a red couch."}

OUTPUT
------
A single CSV: per_caption_chair.csv
    model, method, caption_id, CHAIRs_binary, num_hallucinated,
    num_mentioned, caption_length_words


HOW TO RUN
==========

Step 0 — One-time setup (download COCO annotations):
    mkdir -p coco_annotations
    cd coco_annotations
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip annotations_trainval2014.zip
    cd ..

Step 1 — Organize your caption files in a directory:
    captions/
      llava_vcd.jsonl
      llava_opera.jsonl
      llava_baseline.jsonl
      internvl_vcd.jsonl
      ...

    Naming convention: {model}_{method}.jsonl
    (model and method must match the names in chair_eval_all.csv:
     instructblip, internvl, llava, minicpm, qwen
     and methods: baseline, agla, causalmm, clearsight, halc, icd, opera, pai, sid, vcd, deco)

Step 2 — Run this script:
    python3 per_caption_chair_eval.py \
        --caption_dir captions/ \
        --coco_dir coco_annotations/ \
        --output per_caption_chair.csv \
        --split val

    If you want a faster subset for testing:
    python3 per_caption_chair_eval.py \
        --caption_dir captions/ \
        --coco_dir coco_annotations/ \
        --output per_caption_chair.csv \
        --split val \
        --max_images 100


AFTER THIS
==========
Run the matched-length truncation analysis:
    python3 matched_length_truncation.py per_caption_chair.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

# ── MSCOCO synonym map (standard CHAIR synonym dictionary) ──
# Maps word variants to canonical MSCOCO category names.
# Abridged version covering the most common COCO categories.
SYNONYM_MAP = {
    # person
    "person": "person", "people": "person", "man": "person", "men": "person",
    "woman": "person", "women": "person", "child": "person", "children": "person",
    "guy": "person", "girl": "person", "boy": "person", "lady": "person",
    "gentleman": "person", "human": "person", "baby": "person", "kid": "person",
    # bicycle
    "bicycle": "bicycle", "bike": "bicycle", "cycle": "bicycle",
    # car
    "car": "car", "automobile": "car", "vehicle": "car", "sedan": "car",
    # motorcycle
    "motorcycle": "motorcycle", "motorbike": "motorcycle",
    # airplane
    "airplane": "airplane", "plane": "airplane", "aeroplane": "airplane",
    # bus
    "bus": "bus", "autobus": "bus",
    # train
    "train": "train", "locomotive": "train",
    # truck
    "truck": "truck", "lorry": "truck", "pickup": "truck",
    # boat
    "boat": "boat", "ship": "boat", "vessel": "boat", "sailboat": "boat",
    # traffic light
    "traffic light": "traffic light", "stoplight": "traffic light",
    # fire hydrant
    "fire hydrant": "fire hydrant", "hydrant": "fire hydrant",
    # stop sign
    "stop sign": "stop sign", "stopsign": "stop sign",
    # parking meter
    "parking meter": "parking meter", "meter": "parking meter",
    # bench
    "bench": "bench", "seat": "bench", "pew": "bench",
    # bird
    "bird": "bird", "birds": "bird",
    # cat
    "cat": "cat", "cats": "cat", "kitten": "cat",
    # dog
    "dog": "dog", "dogs": "dog", "puppy": "dog",
    # horse
    "horse": "horse", "horses": "horse",
    # sheep
    "sheep": "sheep", "ewe": "sheep", "lamb": "sheep",
    # cow
    "cow": "cow", "cattle": "cow", "bull": "cow", "calf": "cow",
    # elephant
    "elephant": "elephant", "elephants": "elephant",
    # bear
    "bear": "bear", "bears": "bear",
    # zebra
    "zebra": "zebra", "zebras": "zebra",
    # giraffe
    "giraffe": "giraffe", "giraffes": "giraffe",
    # backpack
    "backpack": "backpack", "back pack": "backpack", "rucksack": "backpack",
    # umbrella
    "umbrella": "umbrella", "brolly": "umbrella",
    # handbag
    "handbag": "handbag", "purse": "handbag", "bag": "handbag",
    # tie
    "tie": "tie", "necktie": "tie", "neck tie": "tie",
    # suitcase
    "suitcase": "suitcase", "luggage": "suitcase",
    # frisbee
    "frisbee": "frisbee",
    # skis
    "skis": "skis", "ski": "skis",
    # snowboard
    "snowboard": "snowboard", "snow board": "snowboard",
    # sports ball
    "sports ball": "sports ball", "ball": "sports ball",
    "baseball": "sports ball", "basketball": "sports ball",
    "soccer ball": "sports ball", "tennis ball": "sports ball",
    "football": "sports ball", "volleyball": "sports ball",
    # kite
    "kite": "kite",
    # baseball bat
    "baseball bat": "baseball bat", "bat": "baseball bat",
    # baseball glove
    "baseball glove": "baseball glove", "glove": "baseball glove",
    # skateboard
    "skateboard": "skateboard", "skate board": "skateboard",
    # surfboard
    "surfboard": "surfboard", "surf board": "surfboard",
    # tennis racket
    "tennis racket": "tennis racket", "racket": "tennis racket", "racquet": "tennis racket",
    # bottle
    "bottle": "bottle", "bottles": "bottle",
    # wine glass
    "wine glass": "wine glass", "wineglass": "wine glass", "glass": "wine glass",
    # cup
    "cup": "cup", "mug": "cup", "teacup": "cup",
    # fork
    "fork": "fork", "forks": "fork",
    # knife
    "knife": "knife", "knives": "knife",
    # spoon
    "spoon": "spoon", "spoons": "spoon",
    # bowl
    "bowl": "bowl", "bowls": "bowl",
    # banana
    "banana": "banana", "bananas": "banana",
    # apple
    "apple": "apple", "apples": "apple",
    # sandwich
    "sandwich": "sandwich", "sandwiches": "sandwich",
    # orange
    "orange": "orange", "oranges": "orange",
    # broccoli
    "broccoli": "broccoli",
    # carrot
    "carrot": "carrot", "carrots": "carrot",
    # hot dog
    "hot dog": "hot dog", "hotdog": "hot dog",
    # pizza
    "pizza": "pizza", "pizzas": "pizza",
    # donut
    "donut": "donut", "doughnut": "donut",
    # cake
    "cake": "cake", "cakes": "cake",
    # chair
    "chair": "chair", "chairs": "chair", "seat": "chair",
    # couch
    "couch": "couch", "sofa": "couch", "settee": "couch",
    # potted plant
    "potted plant": "potted plant", "plant": "potted plant",
    # bed
    "bed": "bed", "beds": "bed",
    # dining table
    "dining table": "dining table", "table": "dining table",
    # toilet
    "toilet": "toilet", "commode": "toilet",
    # tv
    "tv": "tv", "television": "tv", "telly": "tv",
    # laptop
    "laptop": "laptop", "notebook": "laptop",
    # mouse
    "mouse": "mouse", "mice": "mouse",
    # remote
    "remote": "remote", "remote control": "remote",
    # keyboard
    "keyboard": "keyboard",
    # cell phone
    "cell phone": "cell phone", "phone": "cell phone", "mobile": "cell phone",
    "smartphone": "cell phone",
    # microwave
    "microwave": "microwave", "microwave oven": "microwave",
    # oven
    "oven": "oven", "stove": "oven",
    # toaster
    "toaster": "toaster",
    # sink
    "sink": "sink",
    # refrigerator
    "refrigerator": "refrigerator", "fridge": "refrigerator",
    # book
    "book": "book", "books": "book",
    # clock
    "clock": "clock", "timepiece": "clock",
    # vase
    "vase": "vase", "vases": "vase",
    # scissors
    "scissors": "scissors",
    # teddy bear
    "teddy bear": "teddy bear", "teddy": "teddy bear",
    # hair drier
    "hair drier": "hair drier", "hair dryer": "hair drier", "dryer": "hair drier",
    # toothbrush
    "toothbrush": "toothbrush", "tooth brush": "toothbrush",
}


def load_coco_annotations(coco_dir, split="val"):
    """Load COCO ground-truth object annotations.

    Returns: dict mapping image_id -> set of canonical object names present.
    Combines objects from instance annotations (segmentation) and caption
    annotations (human mentions).
    """
    import_path = os.path.join(coco_dir, f"instances_{split}2014.json")
    captions_path = os.path.join(coco_dir, f"captions_{split}2014.json")

    # Load instance annotations
    with open(import_path) as f:
        instances = json.load(f)

    # Map category_id -> category name
    cat_id_to_name = {}
    for cat in instances["categories"]:
        cat_id_to_name[cat["id"]] = cat["name"].lower().strip()

    # Build image_id -> objects from segmentation masks
    img_to_objects = defaultdict(set)
    for ann in instances["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_id_to_name[ann["category_id"]]
        # Map to canonical name if in our synonym map
        canonical = SYNONYM_MAP.get(cat_name, cat_name)
        img_to_objects[img_id].add(canonical)

    # Load caption annotations to enrich with human-mentioned objects
    with open(captions_path) as f:
        captions = json.load(f)

    for ann in captions["annotations"]:
        img_id = ann["image_id"]
        caption_text = ann["caption"].lower()
        # Extract nouns using simple heuristics (all words are candidates)
        words = re.findall(r'\b[a-z]+\b', caption_text)
        for w in words:
            if w in SYNONYM_MAP:
                img_to_objects[img_id].add(SYNONYM_MAP[w])

    return dict(img_to_objects)


def extract_objects_from_caption(caption):
    """Extract MSCOCO object words from a caption.

    Returns: (words_list, node_words_list, indices_list)
    - words_list: canonical object names found in the caption
    - node_words_list: the original word forms (for debugging)
    - indices_list: word positions in the caption
    """
    text = caption.lower()
    # Tokenize by splitting on non-alphanumeric
    words = re.findall(r'\b[a-z]+\b', text)

    found_objects = []
    node_words = []
    indices = []

    i = 0
    while i < len(words):
        # Check for multi-word objects first (two-word combinations)
        matched = False
        if i + 1 < len(words):
            bigram = words[i] + " " + words[i + 1]
            if bigram in SYNONYM_MAP:
                found_objects.append(SYNONYM_MAP[bigram])
                node_words.append(bigram)
                indices.append(i)
                i += 2
                matched = True
                continue

        # Check single word
        if words[i] in SYNONYM_MAP:
            found_objects.append(SYNONYM_MAP[words[i]])
            node_words.append(words[i])
            indices.append(i)
        i += 1

    return found_objects, node_words, indices


def evaluate_caption(caption, gt_objects):
    """Evaluate one caption against ground-truth objects.

    Returns:
        chairs_binary: 1 if caption contains ≥1 hallucinated object, else 0
        num_hallucinated: count of hallucinated MSCOCO objects
        num_mentioned: count of MSCOCO objects mentioned in caption
        caption_length: word count of caption
    """
    mentioned_objects, _, _ = extract_objects_from_caption(caption)
    num_mentioned = len(mentioned_objects)
    caption_length = len(re.findall(r'\b\w+\b', caption)) if caption.strip() else 0

    if num_mentioned == 0:
        return 0, 0, 0, caption_length

    num_hallucinated = 0
    for obj in mentioned_objects:
        if obj not in gt_objects:
            num_hallucinated += 1

    chairs_binary = 1 if num_hallucinated > 0 else 0
    return chairs_binary, num_hallucinated, num_mentioned, caption_length


def process_caption_file(jsonl_path, model, method, gt_objects, max_images=None):
    """Process one caption JSONL file.

    Returns list of per-caption records.
    """
    records = []
    with open(jsonl_path) as f:
        for line_idx, line in enumerate(f):
            if max_images and line_idx >= max_images:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Warning: skipping malformed JSON at line {line_idx} in {jsonl_path}")
                continue

            image_id = data.get("image_id")
            caption = data.get("caption", "")

            if image_id is None:
                continue

            gt_set = gt_objects.get(image_id, set())
            chairs_bin, num_hall, num_ment, cap_len = evaluate_caption(caption, gt_set)

            records.append({
                "model": model,
                "method": method,
                "caption_id": f"{image_id}_{line_idx}",
                "image_id": image_id,
                "CHAIRs_binary": chairs_bin,
                "num_hallucinated": num_hall,
                "num_mentioned": num_ment,
                "caption_length_words": cap_len,
            })

    return records


def aggregate_check(records):
    """Print aggregate CHAIRs/CHAIRi/Len for comparison with chair_eval_all.csv."""
    by_key = defaultdict(list)
    for r in records:
        by_key[(r["model"], r["method"])].append(r)

    print(f"\n{'='*70}")
    print("Aggregate verification — compare these numbers against chair_eval_all.csv:")
    print(f"{'Model':<15} {'Method':<12} {'N':>5} {'CHAIRs':>8} {'CHAIRi':>8} {'Len':>7}")
    print("-" * 65)

    for (model, method) in sorted(by_key.keys()):
        recs = by_key[(model, method)]
        n = len(recs)
        chairs = sum(r["CHAIRs_binary"] for r in recs) / n * 100
        h_sum = sum(r["num_hallucinated"] for r in recs)
        m_sum = sum(r["num_mentioned"] for r in recs)
        chairi = h_sum / m_sum * 100 if m_sum > 0 else 0
        len_mean = sum(r["caption_length_words"] for r in recs) / n
        print(f"{model:<15} {method:<12} {n:>5} {chairs:>7.1f}% {chairi:>7.1f}% {len_mean:>6.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-caption CHAIR evaluation for matched-length truncation analysis"
    )
    parser.add_argument(
        "--caption_dir", required=True,
        help="Directory containing JSONL files named {model}_{method}.jsonl"
    )
    parser.add_argument(
        "--coco_dir", required=True,
        help="Directory containing COCO instances_*.json and captions_*.json"
    )
    parser.add_argument(
        "--output", default="per_caption_chair.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--split", default="val", choices=["train", "val"],
        help="COCO split to use (default: val)"
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Max captions per file (for testing)"
    )
    args = parser.parse_args()

    # Load COCO annotations
    print(f"Loading COCO {args.split} annotations from {args.coco_dir}...")
    gt_objects = load_coco_annotations(args.coco_dir, args.split)
    print(f"  Loaded ground-truth objects for {len(gt_objects)} images")

    # Find caption files
    caption_dir = args.caption_dir
    if not os.path.isdir(caption_dir):
        print(f"ERROR: caption directory not found: {caption_dir}")
        sys.exit(1)

    jsonl_files = sorted(
        f for f in os.listdir(caption_dir)
        if f.endswith(".jsonl")
    )
    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in {caption_dir}")
        print(f"Files must be named like: llava_vcd.jsonl, internvl_baseline.jsonl, etc.")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} caption files:")
    for f in jsonl_files:
        print(f"  {f}")

    # Process each file
    all_records = []

    for jsonl_file in sorted(jsonl_files):
        # Parse model_method from filename
        stem = jsonl_file.replace(".jsonl", "")
        parts = stem.split("_", 1)
        if len(parts) < 2:
            print(f"  Warning: cannot parse model/method from '{jsonl_file}', skipping")
            continue
        model = parts[0]
        method = parts[1]

        filepath = os.path.join(caption_dir, jsonl_file)
        print(f"\nProcessing {model}/{method} ({filepath})...")
        records = process_caption_file(filepath, model, method, gt_objects,
                                        max_images=args.max_images)
        print(f"  {len(records)} captions evaluated")
        all_records.extend(records)

    # Write CSV
    fieldnames = [
        "model", "method", "caption_id", "image_id",
        "CHAIRs_binary", "num_hallucinated", "num_mentioned",
        "caption_length_words",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nWrote {len(all_records)} per-caption records to {args.output}")

    # Print aggregate verification
    aggregate_check(all_records)

    print(f"\nDone. Now run:")
    print(f"  python3 matched_length_truncation.py {args.output}")


if __name__ == "__main__":
    main()

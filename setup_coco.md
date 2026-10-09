# How to Set Up COCO 2014 Annotations

The `per_caption_chair_eval.py` and `common_utils/chair_eval.py` scripts
need the MSCOCO 2014 annotations to work. Here's how to get them.

## Option A: Download from COCO Website

```bash
mkdir -p coco_annotations
cd coco_annotations

# Download annotations (about 250 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip

cd ..
```

After downloading, the directory structure should look like:

```
coco_annotations/
└── annotations/
    ├── instances_val2014.json
    ├── instances_train2014.json
    ├── captions_val2014.json
    └── captions_train2014.json
```

## Option B: Use Kaggle

If you have a Kaggle account:

```bash
pip install kaggle --break-system-packages
kaggle datasets download -d jeffaudi/ms-coco-2014-annotations
unzip ms-coco-2014-annotations.zip -d coco_annotations/
```

## Verifying

```bash
ls coco_annotations/annotations/instances_val2014.json
# Should exist and be non-empty
```

## Running the Scripts

Once annotations are in place, update the paths in your scripts or
pass them via command-line arguments:

```bash
python3 per_caption_chair_eval.py \
    --caption_dir captions/ \
    --coco_dir coco_annotations/annotations/ \
    --output per_caption_chair.csv \
    --split val
```

```bash
python3 common_utils/chair_eval.py \
    --cap_file path/to/captions.jsonl \
    --coco_path coco_annotations/annotations/ \
    --save_path results/output.json
```

## Note

The COCO 2014 validation images (~6 GB) are also needed if you want to run
the full inference pipeline from scratch (i.e., generate captions yourself
using your VLM of choice).

# Finetuned DINOv2 Vision Transformer for categorizing Google Fonts

A font classification system that identifies 394 font variants across 32 families from rendered text images, using LoRA fine-tuning of DINOv2.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

### 1. Get Google Fonts

```bash
git clone --filter=blob:none --depth 1 https://github.com/google/fonts.git
```

### 2. Generate dataset

```bash
python dataset_generator.py \
    --font_dir <path to google fonts> \
    --out_dir <output folder> \
    --img_size 224 \
    --font_size 1024 \
    --padding 128
```

Uses all CPU cores by default (`--workers N` to override). Generates ~575 training images and 40 test images per font variant with randomized colors, alignment, line wrapping, and Gaussian noise.

### 3. Clean the dataset

```bash
python dataset_cleaner.py <dataset folder>
```

Prints any corrupted image paths for manual inspection.

### 4. Upload dataset to HuggingFace (optional)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli upload-large-folder <user>/<repo> <dataset folder> --repo-type=dataset
```

For large datasets (200k+ files), tar the train/test folders first to avoid API rate limits:

```bash
tar cf train.tar -C <dataset folder> train/
tar cf test.tar -C <dataset folder> test/
HF_HUB_DISABLE_XET=1 huggingface-cli upload <user>/<repo> train.tar train.tar --repo-type=dataset
HF_HUB_DISABLE_XET=1 huggingface-cli upload <user>/<repo> test.tar test.tar --repo-type=dataset
```

### 5. Train the model

**LoRA (default, recommended):**

```bash
python train_model.py \
    --data_dir <dataset folder> \
    --output_dir <output folder> \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1
```

**Baseline comparisons:**

```bash
# Full fine-tuning (all 87.2M params)
python train_model.py --full_finetune --data_dir <data> --output_dir <out> --epochs 100

# Linear probe (classifier head only, 606K params)
python train_model.py --linear_probe --data_dir <data> --output_dir <out> --epochs 20

# CNN baseline (ResNet-50)
python train_model.py --resnet_baseline --data_dir <data> --output_dir <out> --epochs 100
```

### 6. Resume from checkpoint

```bash
python train_model.py \
    --checkpoint <output folder>/checkpoint-2752 \
    --data_dir <dataset folder> \
    --output_dir <output folder> \
    --epochs 100
```

### 7. Upload model to HuggingFace

```bash
python train_model.py \
    --epochs 0 \
    --data_dir <dataset folder> \
    --checkpoint <output folder>/checkpoint-2752 \
    --huggingface_model_name <user>/<repo>
```

### 8. Run inference

```bash
python serve_model.py <model name or path> <image path>
```

## Cloud Training

Runs training end-to-end on a Vast.ai GPU instance: finds a machine, uploads the code, trains, downloads results, and destroys the instance automatically.

**Setup:**

```bash
pip install vastai
vastai set api-key <your key>
```

**Usage:**

```bash
# Run all baselines (linear probe, ResNet-50, LoRA r=4/8/16, full FT)
bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --mode all

# Or individually
bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --mode lora
bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --mode full
bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --mode resnet
```

Options: `--gpu` (default RTX_4090), `--max_price` (default $2/hr), `--batch_size`, `--epochs`, `--output` (default ./cloud_results).

## Evaluation

```bash
python confusion_matrix.py \
    --data_dir <dataset folder> \
    --model <HuggingFace model name or local path>
```

The model's label set must match the dataset's class folders. The script will check label overlap and abort if there's a mismatch.

Produces:
- `figures/confusion_matrix.pdf` — Row-normalized heatmap grouped by font family
- `figures/top_confused_pairs.pdf` — Bar chart of most frequent misclassifications
- `figures/per_family_accuracy.pdf` — Per-family accuracy breakdown
- `figures/tsne_embeddings.pdf` — t-SNE of [CLS] embeddings
- `figures/font_dendrogram.pdf` — UPGMA clustering of font families
- `figures/metrics.tex` — LaTeX macros for paper
- `confusion_matrix.json` — Raw counts
- `bad_images.json` — All misclassified images

## Paper

```bash
# Full build (evaluation + LaTeX)
bash build_paper.sh --data_dir <dataset folder> --model <model>

# LaTeX only (skip evaluation)
bash build_paper.sh --skip-matrix
```

## Handler

`handler.py` implements the preprocessing pipeline (pad-to-square + resize + normalize) used at both training and inference time. It's bundled with the model on HuggingFace for Inference Endpoints.

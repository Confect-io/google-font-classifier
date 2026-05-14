# Finetuned DINOv2 Vision Transformer for categorizing Google Fonts

A font classification system that identifies 394 font variants across 32 families from rendered text images, using LoRA fine-tuning of DINOv2. Achieves 98.9% top-1 validation accuracy with only ~1% of parameters trainable.

## Citation

If you use GoogleFontsBench, the training pipeline, or the pretrained models in your work, please cite the arXiv preprint:

```bibtex
@article{chen2025googlefontsbench,
  title   = {Parameter-Efficient Fine-Tuning of DINOv2 for Large-Scale Font Classification},
  author  = {Chen, Daniel and Lowe, Marcus and Zinn, Zaria},
  journal = {arXiv preprint},
  year    = {2025},
  note    = {GoogleFontsBench benchmark and DINOv2 font classification baselines}
}
```

If an arXiv identifier is available, add it to the BibTeX entry as `eprint`, `archivePrefix`, and `primaryClass`.

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
    --batch_size 64 \
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

Runs training end-to-end on Vast.ai GPU instances: finds a machine, uploads the code, trains, uploads results to HuggingFace, and destroys the instance automatically. Includes auto-retry (up to 5 instances), health checks, and crash log upload.

**Setup:**

```bash
pip install vastai
vastai set api-key <your key>
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
huggingface-cli login
```

**Usage:**

```bash
# Run all baselines on separate instances in parallel
bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --hf_results dchen0/font-model-results --mode all --gpu RTX_3090 --parallel

# Run a single mode
bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --hf_results dchen0/font-model-results --mode lora --gpu RTX_3090

# Dry run (tiny test dataset, validates full pipeline in ~5 min)
bash cloud_train.sh --dry_run --gpu RTX_3090
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--hf_dataset` | (required) | HuggingFace dataset to train on |
| `--hf_results` | (required) | HuggingFace repo for results upload |
| `--mode` | `lora` | Training mode: `lora`, `lora4`, `lora16`, `full`, `linear`, `resnet`, or `all` |
| `--gpu` | `RTX_4090` | GPU type (e.g., `RTX_3090`, `A100`) |
| `--max_price` | `2.00` | Max hourly price in USD |
| `--batch_size` | `64` | Training batch size |
| `--epochs` | `100` | Number of training epochs |
| `--num_gpus` | `1` | GPUs per instance (multi-GPU via `accelerate`) |
| `--parallel` | off | Launch each mode on a separate instance |
| `--dry_run` | off | Use tiny test dataset, 1 epoch, defaults to all modes |
| `--ssh_key` | `~/.ssh/vastai` | SSH key for Vast.ai instances |

**Features:**
- Auto-retry with up to 5 different instances per mode
- Health check after launch (connectivity, CUDA, pip)
- Checkpoints synced to HuggingFace every 10 minutes (resumable on preemption)
- Training logs uploaded on any exit (crash, signal, or success)
- Instance auto-destroys after uploading results

**Dry run:**

Always dry run before a full training run to catch issues early:

```bash
# Test all modes (default)
bash cloud_train.sh --dry_run --gpu RTX_3090

# Test a specific mode
bash cloud_train.sh --dry_run --mode resnet --gpu RTX_3090
```

This uses a tiny test dataset (`dchen0/font_crops_test`, 3 classes, 39 images) to validate the entire pipeline in ~5 minutes.

To regenerate the test dataset:

```bash
python create_test_dataset.py --synthetic --upload
```

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
- `figures/metrics.tex` — LaTeX macros for paper (including SWER with typographic metadata distance)
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

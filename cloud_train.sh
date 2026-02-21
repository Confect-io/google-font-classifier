#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Cloud training setup script
#
# Usage (on a cloud GPU instance):
#   bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --mode lora
#   bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --mode full
#   bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --mode linear
#   bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --mode all
#
# Requirements:
#   - NVIDIA GPU with CUDA
#   - Python 3.10+
#   - HUGGINGFACE_API_KEY env var (for uploading results)
# -----------------------------------------------------------------------
set -e

HF_DATASET=""
MODE="lora"
BATCH_SIZE=64
EPOCHS=100
LR=1e-4
OUTPUT_BASE="./output"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf_dataset)   HF_DATASET="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --batch_size)   BATCH_SIZE="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --output)       OUTPUT_BASE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$HF_DATASET" ]; then
    echo "Usage: bash cloud_train.sh --hf_dataset <HF dataset name> --mode <lora|full|linear|all>"
    exit 1
fi

echo "============================================"
echo "  Cloud Training Setup"
echo "  Dataset:    $HF_DATASET"
echo "  Mode:       $MODE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs:     $EPOCHS"
echo "  LR:         $LR"
echo "============================================"

# --- 1. Install dependencies ---
echo "==> Installing dependencies"
pip install -q torch torchvision transformers datasets peft accelerate \
    safetensors huggingface_hub pillow numpy scikit-learn tensorboard

# --- 2. Clone the repo ---
if [ ! -d "font-model" ]; then
    echo "==> Cloning font-model repo"
    git clone https://github.com/Create-Inc/font-model.git
fi
cd font-model

# --- 3. Download dataset from HuggingFace ---
echo "==> Downloading dataset from HuggingFace: $HF_DATASET"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_DATASET', repo_type='dataset', local_dir='data')
"

# If dataset is packed as tar files, extract them
if [ -f "data/train.tar" ]; then
    echo "==> Extracting train.tar..."
    tar xf data/train.tar -C data/
    rm data/train.tar
fi
if [ -f "data/test.tar" ]; then
    echo "==> Extracting test.tar..."
    tar xf data/test.tar -C data/
    rm data/test.tar
fi

# --- 4. Run training ---
run_training() {
    local mode_name=$1
    local extra_flags=$2
    local output_dir="${OUTPUT_BASE}/${mode_name}"

    echo ""
    echo "============================================"
    echo "  Training: $mode_name"
    echo "  Output:   $output_dir"
    echo "============================================"

    python3 train_model.py \
        --data_dir data \
        --output_dir "$output_dir" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LR" \
        $extra_flags

    echo "==> Finished: $mode_name"
}

case "$MODE" in
    lora)
        run_training "lora_r8" ""
        ;;
    lora4)
        run_training "lora_r4" "--lora_rank 4 --lora_alpha 8"
        ;;
    lora16)
        run_training "lora_r16" "--lora_rank 16 --lora_alpha 32"
        ;;
    full)
        run_training "full_finetune" "--full_finetune"
        ;;
    linear)
        run_training "linear_probe" "--linear_probe"
        ;;
    resnet)
        run_training "resnet50" "--resnet_baseline"
        ;;
    all)
        # Run linear probe first (fastest to converge)
        run_training "linear_probe" "--linear_probe --epochs 20"
        run_training "resnet50" "--resnet_baseline"
        run_training "lora_r4" "--lora_rank 4 --lora_alpha 8"
        run_training "lora_r8" ""
        run_training "lora_r16" "--lora_rank 16 --lora_alpha 32"
        run_training "full_finetune" "--full_finetune"
        ;;
    *)
        echo "Unknown mode: $MODE (use lora, lora4, lora16, full, linear, resnet, or all)"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  All training runs complete!"
echo "  Results in: $OUTPUT_BASE/"
echo "============================================"

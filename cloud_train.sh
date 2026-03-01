#!/usr/bin/env bash
# -----------------------------------------------------------------------
# End-to-end cloud training on Vast.ai
#
# Rents a GPU, uploads the training script, runs training, and
# downloads results. Requires the vastai CLI:
#   pip install vastai
#   vastai set api-key <your key>
#
# Usage:
#   bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --mode lora
#   bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --mode all
#   bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --mode all --gpu A100
#
# When training completes, results are downloaded to ./cloud_results/
# -----------------------------------------------------------------------
set -e

HF_DATASET=""
MODE="lora"
BATCH_SIZE=64
EPOCHS=100
LR=1e-4
GPU="RTX_4090"  # default GPU type
MAX_PRICE=2.00  # max $/hr
DISK_GB=100
OUTPUT_LOCAL="./cloud_results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf_dataset)   HF_DATASET="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --batch_size)   BATCH_SIZE="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --gpu)          GPU="$2"; shift 2 ;;
        --max_price)    MAX_PRICE="$2"; shift 2 ;;
        --output)       OUTPUT_LOCAL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$HF_DATASET" ]; then
    echo "Usage: bash cloud_train.sh --hf_dataset <HF dataset> --mode <lora|full|linear|resnet|all>"
    echo ""
    echo "Options:"
    echo "  --gpu          GPU type (default: RTX_4090). Examples: A100, RTX_4090, RTX_3090"
    echo "  --max_price    Max hourly price in USD (default: 2.00)"
    echo "  --batch_size   Batch size (default: 64)"
    echo "  --epochs       Number of epochs (default: 100)"
    echo "  --mode         Training mode: lora, lora4, lora16, full, linear, resnet, or all"
    echo "  --output       Local directory for results (default: ./cloud_results)"
    exit 1
fi

# Check vastai CLI is installed
if ! command -v vastai &> /dev/null; then
    echo "Error: vastai CLI not found. Install with: pip install vastai"
    echo "Then set your API key: vastai set api-key <your key>"
    exit 1
fi

echo "============================================"
echo "  Cloud Training (Vast.ai)"
echo "  Dataset:    $HF_DATASET"
echo "  Mode:       $MODE"
echo "  GPU:        $GPU"
echo "  Max price:  \$$MAX_PRICE/hr"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs:     $EPOCHS"
echo "============================================"

# --- Build the remote training script ---
REMOTE_SCRIPT=$(cat <<'TRAINING_SCRIPT'
#!/bin/bash
set -e

HF_DATASET="__HF_DATASET__"
MODE="__MODE__"
BATCH_SIZE=__BATCH_SIZE__
EPOCHS=__EPOCHS__
LR=__LR__
OUTPUT_BASE="/workspace/output"

echo "==> Installing dependencies"
pip install -q transformers datasets peft accelerate safetensors huggingface_hub pillow numpy scikit-learn tensorboard fontTools

echo "==> Cloning font-model repo"
cd /workspace
if [ ! -d "font-model" ]; then
    git clone https://github.com/Create-Inc/font-model.git
fi
cd font-model

echo "==> Downloading dataset from HuggingFace: $HF_DATASET"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${HF_DATASET}', repo_type='dataset', local_dir='data')
"

# Extract tar files
for tarfile in data/train*.tar data/test*.tar; do
    if [ -f "$tarfile" ]; then
        echo "==> Extracting $tarfile..."
        tar xf "$tarfile" -C data/
        rm "$tarfile"
    fi
done

echo "==> Dataset ready: $(ls data/train/ | wc -l) train variants, $(ls data/test/ | wc -l) test variants"

# --- Training function ---
run_training() {
    local mode_name=$1
    local extra_flags=$2
    local output_dir="${OUTPUT_BASE}/${mode_name}"
    echo ""
    echo "============================================"
    echo "  Training: $mode_name"
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
    lora)    run_training "lora_r8" "" ;;
    lora4)   run_training "lora_r4" "--lora_rank 4 --lora_alpha 8" ;;
    lora16)  run_training "lora_r16" "--lora_rank 16 --lora_alpha 32" ;;
    full)    run_training "full_finetune" "--full_finetune" ;;
    linear)  run_training "linear_probe" "--linear_probe" ;;
    resnet)  run_training "resnet50" "--resnet_baseline" ;;
    all)
        run_training "linear_probe" "--linear_probe --epochs 20"
        run_training "resnet50" "--resnet_baseline"
        run_training "lora_r4" "--lora_rank 4 --lora_alpha 8"
        run_training "lora_r8" ""
        run_training "lora_r16" "--lora_rank 16 --lora_alpha 32"
        run_training "full_finetune" "--full_finetune"
        ;;
    *) echo "Unknown mode: $MODE"; exit 1 ;;
esac

echo ""
echo "============================================"
echo "  ALL TRAINING COMPLETE"
echo "  Results in: $OUTPUT_BASE/"
echo "============================================"
TRAINING_SCRIPT
)

# Substitute variables into the remote script
REMOTE_SCRIPT="${REMOTE_SCRIPT//__HF_DATASET__/$HF_DATASET}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__MODE__/$MODE}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__BATCH_SIZE__/$BATCH_SIZE}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__EPOCHS__/$EPOCHS}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__LR__/$LR}"

# --- Find and rent a GPU ---
echo ""
echo "==> Searching for $GPU instances under \$$MAX_PRICE/hr..."

OFFER_ID=$(vastai search offers "gpu_name=$GPU rentable=true num_gpus=1 dph<=$MAX_PRICE disk_space>=$DISK_GB cuda_vers>=12.0 inet_down>200" \
    -o 'dph' --raw 2>/dev/null | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if not offers:
    print('NONE')
else:
    print(offers[0]['id'])
")

if [ "$OFFER_ID" = "NONE" ]; then
    echo "Error: No $GPU instances found under \$$MAX_PRICE/hr"
    echo "Try: --gpu RTX_3090 or --max_price 3.00"
    exit 1
fi

echo "==> Found offer $OFFER_ID, creating instance..."
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
    --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
    --disk "$DISK_GB" \
    --ssh \
    --direct \
    --onstart-cmd "echo 'Instance ready'" \
    --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('new_contract', 'UNKNOWN'))
")

echo "==> Instance $INSTANCE_ID created. Waiting for it to start..."

# Wait for instance to be running
for i in $(seq 1 60); do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('actual_status', 'unknown'))
" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "running" ]; then
        break
    fi
    echo "  Status: $STATUS (attempt $i/60)..."
    sleep 10
done

if [ "$STATUS" != "running" ]; then
    echo "Error: Instance failed to start after 10 minutes"
    vastai destroy instance "$INSTANCE_ID"
    exit 1
fi

# Get SSH connection info
SSH_INFO=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
host = data.get('ssh_host', '')
port = data.get('ssh_port', '')
print(f'{host} {port}')
")
SSH_HOST=$(echo "$SSH_INFO" | awk '{print $1}')
SSH_PORT=$(echo "$SSH_INFO" | awk '{print $2}')

echo "==> Instance running at $SSH_HOST:$SSH_PORT"

# Upload and run the training script
echo "==> Uploading training script..."
echo "$REMOTE_SCRIPT" | ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" "cat > /workspace/run_training.sh && chmod +x /workspace/run_training.sh"

echo "==> Starting training (this will take hours)..."
echo "    You can monitor with: ssh -p $SSH_PORT root@$SSH_HOST tail -f /workspace/training.log"
echo ""

ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" \
    "nohup bash /workspace/run_training.sh > /workspace/training.log 2>&1 &"

echo "==> Training launched in background on instance $INSTANCE_ID"
echo ""
echo "  Monitor:  ssh -p $SSH_PORT root@$SSH_HOST tail -f /workspace/training.log"
echo "  SSH in:   ssh -p $SSH_PORT root@$SSH_HOST"
echo "  Download: scp -P $SSH_PORT -r root@$SSH_HOST:/workspace/output $OUTPUT_LOCAL"
echo "  Destroy:  vastai destroy instance $INSTANCE_ID"
echo ""
echo "  IMPORTANT: Remember to destroy the instance when training is done!"
echo "  You are being charged \$$MAX_PRICE/hr until you run:"
echo "    vastai destroy instance $INSTANCE_ID"

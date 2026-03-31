#!/usr/bin/env bash
# -----------------------------------------------------------------------
# End-to-end cloud training on Vast.ai
#
# Rents a GPU, uploads the training script, runs training, and uploads
# results to HuggingFace. The instance auto-destroys when done.
#
# Usage:
#   bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --hf_results dchen0/font-model-results --mode lora
#   bash cloud_train.sh --hf_dataset dchen0/font_crops_v5 --hf_results dchen0/font-model-results --mode all --gpu RTX_3090
# -----------------------------------------------------------------------
set -e

HF_DATASET=""
HF_RESULTS=""
MODE=""
MODE_SET=false
BATCH_SIZE=64
EPOCHS=100
LR=1e-4
GPU="RTX_4090"
MAX_PRICE=2.00
DISK_GB=500
OUTPUT_LOCAL="./cloud_results"
SSH_KEY="$HOME/.ssh/vastai"
MAX_RETRIES=5
DRY_RUN=false
PARALLEL=false
NUM_GPUS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf_dataset)   HF_DATASET="$2"; shift 2 ;;
        --hf_results)   HF_RESULTS="$2"; shift 2 ;;
        --mode)         MODE="$2"; MODE_SET=true; shift 2 ;;
        --batch_size)   BATCH_SIZE="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --gpu)          GPU="$2"; shift 2 ;;
        --max_price)    MAX_PRICE="$2"; shift 2 ;;
        --output)       OUTPUT_LOCAL="$2"; shift 2 ;;
        --ssh_key)      SSH_KEY="$2"; shift 2 ;;
        --dry_run)      DRY_RUN=true; shift ;;
        --parallel)     PARALLEL=true; shift ;;
        --num_gpus)     NUM_GPUS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Dry run overrides
if [ "$DRY_RUN" = "true" ]; then
    HF_DATASET="dchen0/font_crops_test"
    HF_RESULTS="${HF_RESULTS:-dchen0/font-model-dry-run}"
    EPOCHS=1
    DISK_GB=50
    # Default to all modes unless a specific mode was explicitly requested
    if [ "$MODE_SET" = "false" ]; then
        MODE="all"
    fi
    echo "*** DRY RUN MODE — using test dataset, 1 epoch, mode=$MODE ***"
fi

# Default mode if not set
if [ -z "$MODE" ]; then
    MODE="lora"
fi

if [ -z "$HF_DATASET" ] || [ -z "$HF_RESULTS" ]; then
    echo "Usage: bash cloud_train.sh --hf_dataset <HF dataset> --hf_results <HF repo for results> --mode <mode>"
    echo ""
    echo "Options:"
    echo "  --hf_dataset   HuggingFace dataset to train on (required)"
    echo "  --hf_results   HuggingFace repo to upload results to (required, e.g. dchen0/font-model-results)"
    echo "  --gpu          GPU type (default: RTX_4090). Examples: A100, RTX_4090, RTX_3090"
    echo "  --max_price    Max hourly price in USD (default: 2.00)"
    echo "  --batch_size   Batch size (default: 64)"
    echo "  --epochs       Number of epochs (default: 100)"
    echo "  --mode         Training mode: lora, lora4, lora16, full, linear, resnet, or all"
    echo "  --output       Local directory for results (default: ./cloud_results)"
    echo "  --dry_run      Use tiny test dataset, 1 epoch (validates full pipeline)"
    echo "  --parallel     Launch each mode on a separate GPU instance (use with --mode all)"
    exit 1
fi

# Parallel mode: launch each training mode as a separate instance
if [ "$PARALLEL" = "true" ] && [ "$MODE" = "all" ]; then
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    # Build common args (exclude --mode, --parallel, --dry_run)
    COMMON_ARGS="--hf_dataset $HF_DATASET --hf_results $HF_RESULTS --gpu $GPU --max_price $MAX_PRICE --batch_size $BATCH_SIZE --lr $LR --ssh_key $SSH_KEY --num_gpus $NUM_GPUS"
    if [ "$DRY_RUN" = "true" ]; then
        COMMON_ARGS="$COMMON_ARGS --dry_run"
    fi

    echo "============================================"
    echo "  Parallel Training — launching 6 instances"
    echo "============================================"

    ALL_MODES="linear:20 resnet:$EPOCHS lora4:$EPOCHS lora:$EPOCHS lora16:$EPOCHS full:$EPOCHS"
    for mode_spec in $ALL_MODES; do
        mode_name="${mode_spec%%:*}"
        mode_epochs="${mode_spec##*:}"
        echo ""
        echo "==> Launching $mode_name (epochs=$mode_epochs)..."
        bash "$SCRIPT_PATH" $COMMON_ARGS --mode "$mode_name" --epochs "$mode_epochs" &
        sleep 5  # stagger to avoid API rate limits
    done

    echo ""
    echo "============================================"
    echo "  All 6 modes launched in parallel."
    echo "  Each instance will upload results to: $HF_RESULTS"
    echo "  and auto-destroy when done."
    echo "============================================"
    wait
    exit 0
fi

# Check vastai CLI is installed
if ! command -v vastai &> /dev/null; then
    echo "Error: vastai CLI not found. Install with: pip install vastai"
    echo "Then set your API key: vastai set api-key <your key>"
    exit 1
fi

# Read the Vast.ai API key for remote self-destruct
VAST_API_KEY=$(python3 -c "
import os
for p in ['~/.config/vastai/vast_api_key', '~/.vast_api_key']:
    path = os.path.expanduser(p)
    if os.path.exists(path):
        print(open(path).read().strip())
        break
" 2>/dev/null)
if [ -z "$VAST_API_KEY" ]; then
    echo "Error: Could not find Vast.ai API key"
    echo "Run: vastai set api-key <your key>"
    exit 1
fi

# Read the HuggingFace token for uploading results
HF_TOKEN=$(python3 -c "
import os
for p in ['~/.cache/huggingface/token', '~/.huggingface/token']:
    path = os.path.expanduser(p)
    if os.path.exists(path):
        print(open(path).read().strip())
        break
" 2>/dev/null)
if [ -z "$HF_TOKEN" ]; then
    echo "Error: Could not find HuggingFace token"
    echo "Run: huggingface-cli login"
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
echo "  GPUs:       $NUM_GPUS"
echo "  Results to: $HF_RESULTS"
echo "============================================"

# --- Build the remote training script ---
REMOTE_SCRIPT=$(cat <<'TRAINING_SCRIPT'
#!/bin/bash
set -eo pipefail

# Always upload the training log to HF before exit (even on crash)
upload_log() {
    echo "==> Uploading training log to HuggingFace..."
    python3 -c "
from huggingface_hub import HfApi
import os, datetime
api = HfApi(token='__HF_TOKEN__')
api.create_repo('__HF_RESULTS__', repo_type='model', exist_ok=True)
log_path = '/workspace/training.log'
if os.path.exists(log_path):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    api.upload_file(
        path_or_fileobj=log_path,
        path_in_repo=f'logs/__MODE___{ts}.log',
        repo_id='__HF_RESULTS__',
        repo_type='model',
    )
    print(f'Log uploaded as logs/__MODE___{ts}.log')
else:
    print('No training log found')
" 2>/dev/null || echo "==> Log upload failed (non-critical)"
}
trap 'echo "SCRIPT CRASHED at line $LINENO (exit code $?)"; upload_log' ERR

HF_DATASET="__HF_DATASET__"
MODE="__MODE__"
BATCH_SIZE=__BATCH_SIZE__
EPOCHS=__EPOCHS__
LR=__LR__
NUM_GPUS=__NUM_GPUS__
OUTPUT_BASE="/workspace/output"
export HF_TOKEN="__HF_TOKEN__"

# Verify internet connectivity (retry a few times — networking can take a moment after boot)
echo "==> Checking internet connectivity..."
for _try in 1 2 3 4 5; do
    if pip install --dry-run pip > /dev/null 2>&1; then
        echo "==> Internet + pip OK"
        break
    fi
    echo "  No connectivity yet (attempt $_try/5)..."
    sleep 15
done
# Final check
if ! pip install --dry-run pip > /dev/null 2>&1; then
    echo "EARLY_FAIL: pip cannot reach PyPI (network or SSL broken)."
    exit 1
fi

echo "==> Installing dependencies"
# Pin torch to 2.6.x to satisfy transformers >=2.6 requirement while staying compatible with CUDA 12.x drivers
pip install -q "torch>=2.6,<2.7" "torchvision>=0.21,<0.22" --index-url https://download.pytorch.org/whl/cu124
pip install -q transformers datasets peft accelerate safetensors huggingface_hub pillow numpy scikit-learn tensorboard fontTools

# Verify CUDA is available
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    echo "EARLY_FAIL: CUDA not available (driver too old or no GPU detected)."
    exit 1
fi
echo "==> CUDA OK ($(python3 -c 'import torch; print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")' 2>/dev/null))"

# System diagnostics
echo "==> System info:"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "  RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo '?')"
echo "  CPU: $(nproc) cores"
echo "  Disk: $(df -h /workspace 2>/dev/null | tail -1 | awk '{print $2, "total,", $4, "free"}')"

echo "==> Cloning font-model repo"
cd /workspace
if [ ! -d "font-model" ]; then
    git clone https://github.com/Create-Inc/font-model.git
fi
cd font-model

echo "==> Downloading dataset from HuggingFace: $HF_DATASET"
for _dl_try in 1 2 3 4 5; do
    if python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${HF_DATASET}', repo_type='dataset', local_dir='data', token='__HF_TOKEN__')
"; then
        break
    fi
    echo "  Download failed (attempt $_dl_try/5), retrying in 30s..."
    sleep 30
done

# Extract tar files and clean macOS resource fork files
for tarfile in data/train*.tar data/test*.tar; do
    if [ -f "$tarfile" ]; then
        echo "==> Extracting $tarfile..."
        tar xf "$tarfile" -C data/
        rm "$tarfile"
    fi
done
find data/ -name '._*' -delete 2>/dev/null || true

# Clean up HF cache to free disk space
rm -rf /root/.cache/huggingface/hub 2>/dev/null || true

echo "==> Dataset ready: $(ls data/train/ | wc -l) train variants, $(ls data/test/ | wc -l) test variants"
df -h /workspace | tail -1

# Pre-cache the dataset (single process) so multi-GPU doesn't build N copies
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "==> Pre-caching dataset for multi-GPU (single process)..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('imagefolder', data_dir='data')
print(f'Cached: {len(ds[\"train\"])} train, {len(ds[\"test\"])} test')
"
    # Delete raw images now that Arrow cache exists
    rm -rf data/train data/test 2>/dev/null || true
    rm -rf /root/.cache/huggingface/hub 2>/dev/null || true
    echo "==> Cache built, raw images cleaned"
    df -h /workspace | tail -1
fi

# --- Training (disable set -e, each run handles its own errors) ---
set +e
FAILED_RUNS=""
run_training() {
    local mode_name=$1
    local extra_flags=$2
    local output_dir="${OUTPUT_BASE}/${mode_name}"

    # Skip if already completed (has a trainer_state.json from a previous run)
    if [ -f "${output_dir}/trainer_state.json" ] || [ -f "${output_dir}/training_args.bin" ]; then
        echo "==> Skipping $mode_name (already completed)"
        return 0
    fi

    echo ""
    echo "============================================"
    echo "  Training: $mode_name (GPUs: $NUM_GPUS)"
    echo "============================================"

    local train_cmd="python3 train_model.py"
    if [ "$NUM_GPUS" -gt 1 ]; then
        train_cmd="accelerate launch --num_processes=$NUM_GPUS --mixed_precision=fp16 train_model.py"
    fi

    if $train_cmd \
        --data_dir data \
        --output_dir "$output_dir" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LR" \
        $extra_flags; then
        echo "==> Finished: $mode_name"
    else
        echo "==> FAILED: $mode_name (exit code $?)"
        FAILED_RUNS="$FAILED_RUNS $mode_name"
    fi
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
if [ -n "$FAILED_RUNS" ]; then
    echo "  TRAINING COMPLETE (with failures)"
    echo "  Failed runs:$FAILED_RUNS"
else
    echo "  ALL TRAINING COMPLETE"
fi
echo "  Results in: $OUTPUT_BASE/"
echo "============================================"

# Upload results to HuggingFace (skip if no output)
if [ -d "$OUTPUT_BASE" ] && [ "$(ls -A $OUTPUT_BASE 2>/dev/null)" ]; then
    echo "==> Uploading results to HuggingFace: __HF_RESULTS__"
    python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='__HF_TOKEN__')
api.create_repo('__HF_RESULTS__', repo_type='model', exist_ok=True)
api.upload_folder(folder_path='$OUTPUT_BASE', repo_id='__HF_RESULTS__', repo_type='model')
print('Upload complete.')
"
else
    echo "==> No results to upload (all runs failed or no output produced)"
fi

# Upload training log (always, even if training failed)
upload_log

# Self-destruct: destroy this instance via Vast.ai API
echo "==> Auto-destroying instance __INSTANCE_ID__..."
curl -s -X PUT "https://console.vast.ai/api/v0/instances/__INSTANCE_ID__/" \
    -H "Authorization: Bearer __VAST_API_KEY__" \
    -H "Content-Type: application/json" \
    -d '{"state": "stopped"}' || true
curl -s -X DELETE "https://console.vast.ai/api/v0/instances/__INSTANCE_ID__/" \
    -H "Authorization: Bearer __VAST_API_KEY__" || true
echo "==> Instance destroyed."
TRAINING_SCRIPT
)

# Substitute variables into the remote script (instance ID/API key done per-attempt)
REMOTE_SCRIPT="${REMOTE_SCRIPT//__HF_DATASET__/$HF_DATASET}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__MODE__/$MODE}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__BATCH_SIZE__/$BATCH_SIZE}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__EPOCHS__/$EPOCHS}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__LR__/$LR}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__NUM_GPUS__/$NUM_GPUS}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__HF_RESULTS__/$HF_RESULTS}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__HF_TOKEN__/$HF_TOKEN}"

# -----------------------------------------------------------------------
# Retry loop: try up to MAX_RETRIES instances until one sticks
# -----------------------------------------------------------------------
TRIED_OFFERS=""
LOG_FILE="cloud_train_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log "Cloud training started at $(date)"
log "Config: dataset=$HF_DATASET mode=$MODE gpu=$GPU max_price=$MAX_PRICE"
log ""

for ATTEMPT in $(seq 1 "$MAX_RETRIES"); do
    log ""
    log "========== Attempt $ATTEMPT/$MAX_RETRIES ($(date)) =========="

    # --- Find a GPU, skipping previously failed offers ---
    log "==> Searching for $GPU instances under \$$MAX_PRICE/hr..."

    OFFER_ID=$(vastai search offers "gpu_name=$GPU rentable=true num_gpus=$NUM_GPUS dph<=$MAX_PRICE disk_space>=$DISK_GB cuda_vers>=12.4 inet_down>500 inet_up>200 reliability>0.95 direct_port_count>=1" \
        -o 'reliability2-' --raw 2>/dev/null | python3 -c "
import json, sys
skip = set('$TRIED_OFFERS'.split())
offers = json.load(sys.stdin)
for o in offers:
    if str(o['id']) not in skip:
        print(o['id'])
        break
else:
    print('NONE')
")

    if [ "$OFFER_ID" = "NONE" ]; then
        log "Error: No more $GPU instances available under \$$MAX_PRICE/hr"
        log "Try: --gpu RTX_3090 or --max_price 3.00"
        exit 1
    fi

    TRIED_OFFERS="$TRIED_OFFERS $OFFER_ID"
    log "==> Found offer $OFFER_ID, creating instance..."

    INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
        --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
        --disk "$DISK_GB" \
        --ssh \
        --direct \
        --onstart-cmd "echo 'Instance ready'" \
        --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('new_contract', 'UNKNOWN'))
")

    log "==> Instance $INSTANCE_ID created. Waiting for it to start..."

    # Wait for instance to be running
    STARTED=false
    for i in $(seq 1 60); do
        STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('actual_status', 'unknown'))
" 2>/dev/null || echo "unknown")
        if [ "$STATUS" = "running" ]; then
            STARTED=true
            break
        fi
        log "  Status: $STATUS (attempt $i/60)..."
        sleep 10
    done

    if [ "$STARTED" = "false" ]; then
        log "==> Instance failed to start. Destroying and retrying..."
        vastai destroy instance "$INSTANCE_ID" 2>/dev/null || true
        continue
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

    log "==> Instance running at $SSH_HOST:$SSH_PORT"

    # Wait for SSH to become available
    log "==> Waiting for SSH..."
    SSH_OK=false
    for i in $(seq 1 30); do
        if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$SSH_PORT" "root@$SSH_HOST" "echo ok" &>/dev/null; then
            SSH_OK=true
            break
        fi
        log "  SSH not ready (attempt $i/30)..."
        sleep 10
    done

    if [ "$SSH_OK" = "false" ]; then
        log "==> SSH never came up. Destroying and retrying..."
        vastai destroy instance "$INSTANCE_ID" 2>/dev/null || true
        continue
    fi

    # Substitute instance-specific variables
    ATTEMPT_SCRIPT="${REMOTE_SCRIPT//__INSTANCE_ID__/$INSTANCE_ID}"
    ATTEMPT_SCRIPT="${ATTEMPT_SCRIPT//__VAST_API_KEY__/$VAST_API_KEY}"

    # Upload and launch training
    log "==> Uploading training script..."
    echo "$ATTEMPT_SCRIPT" | ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" "cat > /workspace/run_training.sh && chmod +x /workspace/run_training.sh"

    log "==> Launching training in background..."
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" \
        "nohup bash /workspace/run_training.sh > /workspace/training.log 2>&1 &"

    # Wait 3 minutes, then check if the script is still running
    # (allows time for connectivity retries + pip install)
    log "==> Waiting 3 minutes to verify instance is healthy..."
    sleep 180

    HEALTH_EXIT=0
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$SSH_PORT" "root@$SSH_HOST" \
        "pgrep -f run_training.sh > /dev/null" 2>/dev/null || HEALTH_EXIT=$?

    if [ "$HEALTH_EXIT" -eq 0 ]; then
        # Grab the remote log so far for the local log
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$SSH_PORT" "root@$SSH_HOST" \
            "cat /workspace/training.log" >> "$LOG_FILE" 2>/dev/null || true

        log ""
        log "============================================"
        log "  Training running on instance $INSTANCE_ID"
        log ""
        log "  When done, results upload to: $HF_RESULTS"
        log "  Then the instance auto-destroys."
        log ""
        log "  Monitor:  ssh -i $SSH_KEY -p $SSH_PORT root@$SSH_HOST tail -f /workspace/training.log"
        log "  SSH in:   ssh -i $SSH_KEY -p $SSH_PORT root@$SSH_HOST"
        log "  Local log: $LOG_FILE"
        log "============================================"
        exit 0
    else
        # Capture the remote log for debugging
        log "==> Training died early on instance $INSTANCE_ID (health exit=$HEALTH_EXIT):"
        log "--- Remote log from instance $INSTANCE_ID ---"
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$SSH_PORT" "root@$SSH_HOST" \
            "cat /workspace/training.log" 2>/dev/null | tee -a "$LOG_FILE" || true
        log "--- End remote log ---"
        log "==> Destroying and retrying..."
        vastai destroy instance "$INSTANCE_ID" 2>/dev/null || true
        continue
    fi
done

log ""
log "Error: Failed to launch training after $MAX_RETRIES attempts."
log "Tried offers: $TRIED_OFFERS"
log "Full log saved to: $LOG_FILE"
exit 1

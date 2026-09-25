# Confect Font Classifier — Training Notes

This is a fork-ish clone of `Create-Inc/font-model` adapted to train the
design-agent's font classifier on a curated subset of popular Google
Fonts. Upstream README is still the authoritative reference for
`train_model.py` flags and `cloud_train.sh` options — this doc only
covers what's specific to our setup and what we changed.

## Why we're retraining

The original `font-classifier-v4` was trained on only ~33 families.
20 of those don't exist in Confect's `fonts` table, so the design-agent's
matched-font hits stayed low even when the classifier was confident.
The goal of this retrain is **broader family coverage with intentional
weight curation** — fewer classes, fonts customers are actually likely
to see, restricted to the weights that matter for ad copy.

We pick fonts by **Google Fonts popularity**, not Confect customer
usage, because we don't have a reliable signal for the latter.

## What we changed vs. upstream

| File | Change | Why |
|---|---|---|
| `curate_fonts.py` | New script | Pulls popularity-sorted Google Fonts metadata, extracts specific weight instances from variable fonts via fontTools, writes static `.ttf` files into `./fonts/`. Replaces the manual font-curation step in upstream's README. |
| `dataset_generator.py:24` | `FONT_ALLOWLIST` replaced | Was 32 hardcoded families. Now holds the 147 family stems produced by `curate_fonts.py`. |
| `cloud_train.sh` | Remote destroy switched from raw `curl` to `vastai destroy instance -y` | The bare `curl -X DELETE` against Vast's REST API silently failed in our dry-runs, leaving zombie instances billing. The CLI works against the same API key file; `-y` is required because without a TTY the interactive `[y/N]` prompt defaults to "no" and the destroy aborts silently. The remote script installs the CLI before training and calls `vastai destroy instance ID -y` at the end. |
| `cloud_train.sh` (`run_training`) | Strip `--linear_probe` / `--resnet_baseline` / `--full_finetune` from `$extra_flags` when resuming | `train_model.py` rejects those flags combined with `--checkpoint` as mutually exclusive. Without the strip, every baseline run crashes immediately when the script finds an existing checkpoint in the HF results repo and tries to resume. |

Nothing else is patched. The repo's `dataset_generator.py` filter and
`train_model.py` flow are untouched.

## HuggingFace repos

| Repo | Purpose | State |
|---|---|---|
| `confect/google-font-dataset` | Training data (`train.tar` + `test.tar`) | Populated, 163,590 images, 266 classes |
| `confect/google-font-classifier` | Trained model checkpoints + ONNX | Empty until first `cloud_train.sh` run |
| `confect/font-classifier-dryrun` | Sandbox for `--dry_run` validations | Throwaway, safe to delete after pipeline checks |

The dataset is 2 commits (one per tar). Tars are required by
`cloud_train.sh` — it expects `data/train*.tar` and `data/test*.tar`
in the HF snapshot and extracts them on the Vast box. See the README's
"Upload dataset to HuggingFace" section for the rationale (HF API
rate-limits a per-file upload at 163k files).

## Vast.ai setup (one-time per workstation)

```bash
uv tool install vastai
vastai set api-key <key from https://cloud.vast.ai/account/>
vastai create ssh-key "$(cat ~/.ssh/id_rsa.pub)"  # or whichever key you want Vast to accept
```

Vast is prepay-only — load credit in the account before kicking off
training. $25 covers a dry-run plus a full LoRA r=8 run on an RTX 3090
with comfortable headroom. HuggingFace itself needs no billing for
public dataset/model repos.

## Bash version gotcha

`cloud_train.sh` needs bash 4+. macOS ships bash 3.2 at `/bin/bash`,
which cannot even PARSE the script (syntax error at the `case "$MODE"`
block, ~line 431).

**Invoke it as `/opt/homebrew/bin/bash cloud_train.sh ...`** — the full
path, explicitly.

An earlier version of this note said to run `./cloud_train.sh` and let the
`#!/usr/bin/env bash` shebang find Homebrew's bash on PATH. That is WRONG on
this machine: `/bin` precedes `/opt/homebrew/bin` in PATH (in both the login
shell and non-interactive shells), so `env bash` resolves to 3.2 — exactly
the version the warning is about. Verify with:

    /usr/bin/env bash --version    # 3.2.57 here, not 5.x

## Never edit cloud_train.sh while a run is in flight

Bash reads a script incrementally by byte offset, so inserting or removing
lines in a running script can make it execute garbage from a stale position.
Wait for the launcher to exit (it exits as soon as the health check passes),
or copy it aside and edit the copy.

Note the launcher and the training job are independent: the remote script is
`nohup`-ed, so `pkill -f cloud_train.sh` stops the launcher WITHOUT stopping
training. That is the move when the launcher is misbehaving — it also stops
it destroying a healthy instance on a bad probe.

## Dry runs give a false negative

`cloud_train.sh` launches training, sleeps 180s, then SSHes in and checks the
job is still alive. If it is not, the script assumes the job died, destroys
the instance and retries — up to 5 times.

A dry run (39 images, 1 epoch) **finishes in ~90 seconds** and the remote
script self-destructs the instance on success. The health check then finds a
dead host and misreads that success as failure, renting four more instances
for nothing.

So: when a dry run looks stuck at "Waiting 3 minutes to verify instance is
healthy", check HuggingFace rather than the local log —

    logs/lora16_<timestamp>.log and <mode>/result_model/ in --hf_results

If those are there, it worked; kill the local script (`pkill -f
cloud_train.sh`) before it starts retrying, and confirm with
`vastai show instances` that nothing is left billing.

Real runs are unaffected: at 153 classes the Arrow cache build alone takes
~2 hours, so the job is always alive at the 3-minute mark.

## End-to-end retrain workflow

### 1. (Optional) Refresh `google_fonts_repo`
```bash
cd ~/Confect/misc/google_fonts_repo && git pull --ff-only
```
Only needed when you want fonts that landed in `google/fonts` after the
last clone. The clone is ~4 GB.

### 2. Curate fonts
```bash
cd ~/Confect/misc/google-font-classifier
uv run --with fonttools python3 curate_fonts.py \
    --google_fonts_repo ../google_fonts_repo \
    --top_n 150 --weights 400 700 \
    --out_dir ./fonts \
    --allowlist_out ./FONT_ALLOWLIST.py
```
Reads Google's public metadata endpoint (no API key) for popularity
ranking, filters to Latin-primary Sans/Serif/Display, and uses
fontTools to instance the variable fonts to the chosen weights.
Outputs static `.ttf` files named `{FamilyStem}-{Weight}.ttf` and a
`FONT_ALLOWLIST.py` literal.

### 3. Paste the allowlist
Copy the contents of `FONT_ALLOWLIST.py` over the `FONT_ALLOWLIST = [...]`
block in `dataset_generator.py` (around line 24).

### 4. Generate the dataset
```bash
uv run --with numpy --with pillow --with fonttools --with tqdm \
    python3 dataset_generator.py --font_dir ./fonts --out_dir ./data --img_size 256
```
~7 minutes on an M-series Mac with 8 worker processes. Produces 575
train + 40 test images per class. Output ~40 GB on disk for ~270 classes.

### 5. Sanity-check + tar
```bash
uv run --with numpy --with pillow --with tqdm python3 dataset_cleaner.py ./data
tar cf train.tar -C ./data train/
tar cf test.tar  -C ./data test/
```

### 6. Upload to HuggingFace
```bash
HF_HUB_DISABLE_XET=1 hf upload confect/google-font-dataset train.tar train.tar --repo-type=dataset
HF_HUB_DISABLE_XET=1 hf upload confect/google-font-dataset test.tar  test.tar  --repo-type=dataset
```
First-time HF auth: `hf auth login` with a write token.

### 7. Dry-run on Vast.ai
```bash
./cloud_train.sh --dry_run --gpu RTX_3090 \
    --hf_results confect/font-classifier-dryrun \
    --ssh_key ~/.ssh/id_rsa
```
Validates the full pipeline (Vast rental, SSH, HF download, training
start, results upload, instance teardown) on a tiny test dataset in
~5 min for ~$0.05.

### 8. Real training run
```bash
./cloud_train.sh \
    --hf_dataset confect/google-font-dataset \
    --hf_results confect/google-font-classifier \
    --mode lora \
    --gpu RTX_3090 \
    --epochs 100 \
    --max_price 0.50 \
    --ssh_key ~/.ssh/id_rsa
```
- `--mode lora` = LoRA r=8 (paper's production config).
- Expected time: ~7-10 hours on RTX 3090 for 266 classes (paper's full
  394-class run was 33 hrs). Cost: ~$3-5 at $0.30-0.50/hr.
- Auto-retries up to 5 instances on failure; destroys the box when done.

## Font-weight benchmark

`weight_probe.py` is the promotion gate for deriving upright CSS weights
300–900 from OCR crops after the font family is known. It verifies filename
weights against each font's OS/2 metadata, intersects variants with an optional
Confect catalogue export, uses disjoint calibration/validation/final seeds, and
reports confusion matrices, per-family/category/background results,
precision/coverage curves, and abstention reasons.

The locked 2026-09-24 oracle-family run covered all 153 classifier families and
all 683 available upright variants. Single-weight families were excluded from
the gate. Its 12,840-crop final set contained 20 examples per multi-weight
family/weight pair, balanced across flat, gradient, textured, and photographic
backgrounds.

| Result | Locked final |
|---|---:|
| Multi-weight families | 112 |
| Measurable coverage | 71.3% |
| Exact accuracy on measurable crops | 44.7% |
| Authoritative precision / coverage | 0.0% / 0.0% |
| Legacy global-bold accuracy | 59.7% |
| Legacy emitted-bold precision | 69.7% |

This fails the original 95% exact precision at 50% coverage even when the true
family is supplied, so the result must not be treated as authoritative.
A fixed 20-family same-text diagnostic reached 100% only when it could render
the exact source text with the same font files and render parameters; that is a
diagnostic upper bound, not a deployable method.

After the locked run, the product tolerance was clarified as three practical
groups: 300/400, 500/600, and 700/800/900, with a result within 100 also useful.
Rescoring the saved final predictions did not require another image run:

| Relaxed result | Measurable crops |
|---|---:|
| Top prediction within 100 | 82.2% |
| Top prediction in the requested group | 69.7% |
| Either of the top two within 100 | 91.7% |
| Either of the top two in the requested group | 84.0% |

An exploratory disjoint 40/60 split selected a raw-confidence threshold on the
first partition and achieved 95.9% precision at 51.9% coverage on the second
for “one of the two candidates is within 100.” This criterion was defined only
after inspecting the original final result, and the algorithm still cannot
authoritatively choose between the candidates. Treat it as a promising advisory
experiment that needs a newly locked confirmation set, not as a promotion.

The shipped advisory policy is deliberately stricter than the raw two-candidate
result. `weight_probe.py` exports family centroids, supported variants,
probability bins, and a validation-selected threshold. Runtime also rejects any
bin calibrated below 90% precision. Applied unchanged to the locked final set,
that policy reached 93.5% requested-group precision at 21.4% coverage. The
design agent presents the two supported candidates on the matching OCR line and
requires a visual choice; rejected crops receive no estimate. It does not claim
an authoritative weight.

## Family + weight model v2

The implementation and promotion plan is in `FONT_WEIGHT_MODEL_V2.md`. V2 uses
the same DINOv2 backbone with a 153-family head and a separate two-logit ordinal
head for 300/400, 500/600, and 700/800/900. Same-text weight pairs also train
the family output to remain stable when only boldness changes.

Run the local end-to-end smoke test before renting a GPU:

```bash
uv run --with numpy --with pillow --with fonttools --with tqdm \
  python3 dataset_generator.py \
    --font_dir ./fonts_v2 --out_dir ./.data_out/font-weight-v2-smoke \
    --families Montserrat Lora --train_per_class 8 --test_per_class 4 \
    --workers 1 --seed 42

uv run python3 split_multitask_evaluation.py \
  ./.data_out/font-weight-v2-smoke

uv run --with 'torch>=2.6,<2.7' --with 'torchvision>=0.21,<0.22' \
  --with 'transformers<5' --with peft --with accelerate \
  --with safetensors --with tensorboard --with pillow --with numpy \
  --with fonttools python3 train_multitask.py \
    --data_dir ./.data_out/font-weight-v2-smoke \
    --labels ./font_labels_v6.json \
    --initial_adapter confect/google-font-classifier-v6 \
    --initial_adapter_subfolder lora_r16/result_model \
    --output_dir ./.data_out/font-weight-v2-output \
    --batch_size 2 --epochs 1

uv run --with 'torch>=2.6,<2.7' --with 'torchvision>=0.21,<0.22' \
  --with 'transformers<5' --with peft --with safetensors --with onnx \
  --with pillow --with numpy python3 export_multitask_onnx.py \
    --adapter ./.data_out/font-weight-v2-output/result_model \
    --onnx_out ./.data_out/font-classifier-v7-weight-smoke.onnx

uv run --with 'torch>=2.6,<2.7' --with 'torchvision>=0.21,<0.22' \
  --with 'transformers<5' --with peft --with safetensors --with onnxruntime \
  --with pillow --with numpy python3 verify_multitask_onnx.py \
    --adapter ./.data_out/font-weight-v2-output/result_model \
    --onnx ./.data_out/font-classifier-v7-weight-smoke.onnx \
    --image <one-generated-test-jpg>
```

Each evaluation epoch logs family top-1/top-5, family accuracy by source
weight group, weight-group accuracy, group-distance error, joint accuracy,
paired-family stability, and the family/weight/consistency losses. The Vast
launcher captures these in the normal uploaded training log.

The full dataset lives beside v6 rather than replacing it:

```bash
uv run --with numpy --with pillow --with fonttools --with tqdm \
  python3 dataset_generator.py \
    --font_dir ./fonts_v2 --out_dir ./data_v2 --img_size 256 --seed 42

uv run python3 split_multitask_evaluation.py ./data_v2

tar cf train-v2.tar -C ./data_v2 train/
tar cf test-v2.tar -C ./data_v2 validation/ test/
HF_HUB_DISABLE_XET=1 hf upload confect/google-font-weight-dataset-v2 \
  train-v2.tar train.tar --repo-type=dataset
HF_HUB_DISABLE_XET=1 hf upload confect/google-font-weight-dataset-v2 \
  test-v2.tar test.tar --repo-type=dataset
```

Do not start the Vast run until the local checkpoint has been reloaded, exported
with `export_multitask_onnx.py`, and executed through ONNX Runtime.

The multitask adapter is relative to the merged v6 adapter, not raw DINOv2.
`export_multitask_onnx.py` reconstructs v6 before applying the new adapter, and
the exported metadata records that dependency. When exporting an intermediate
checkpoint, whose directory has no metadata of its own, pass the result model's
metadata explicitly:

```bash
uv run --with 'torch>=2.6,<2.7' --with 'torchvision>=0.21,<0.22' \
  --with 'transformers<5' --with peft --with onnx python3 \
  export_multitask_onnx.py \
    --adapter <checkpoint-directory> \
    --metadata <result-model>/font_model_metadata.json \
    --onnx_out <output>.onnx
```

When approved, the distinct v2 launch command is:

```bash
PATH="/opt/homebrew/bin:$PATH" ./cloud_train.sh \
  --hf_dataset confect/google-font-weight-dataset-v2 \
  --hf_results confect/google-font-classifier-v7-weight \
  --mode multitask --gpu RTX_4090 --batch_size 32 --epochs 100
```

The launcher requires these exact v2 dataset and result repository names and
rejects datasets without train, validation, and locked-test metadata, so
passing the old v6 dataset fails before training rather than silently training
the wrong objective. Checkpoint selection uses validation only; the locked test
is evaluated once after training. Multitask training initializes the family path from
`confect/google-font-classifier-v6/lora_r16/result_model`, then adds and trains
the new ordinal head.

## Swapping the trained model into the design-agent

After training finishes, the new checkpoint sits in
`confect/google-font-classifier-v7-weight`.
To put it in production:

1. Download the checkpoint, convert to ONNX (export
   `Dinov2ForImageClassification` via `torch.onnx.export` — same as v4).
2. Upload to `cdn.confect.io/static/font-classifier-vN.onnx`.
3. In `confect_internal/ai_tooling/design_agent/design_agent/`:
   - Update `_DEFAULT_MODEL_URL` in `src/design_agent/ocr/fonts.py`.
   - Update `ADD --checksum=sha256:...` for the new file in `Dockerfile`.
   - Replace `src/design_agent/ocr/font_labels.json` with the new
     id-to-label map produced by training.
4. Re-run `scripts/build_font_db_mapping.py` against an up-to-date
   `fonts` CSV dump to regenerate `font_db_mapping.json` for whatever
   new families the model now recognises.
5. Validate with `pytest tests/test_fonts.py -v` against our real-ad
   fixtures (these are out-of-distribution from the training data, so
   they're the meaningful acceptance signal — paper's 99% top-1 is
   in-distribution and not predictive of real performance).

## Useful upstream references

- Hyperparameters and accuracy numbers: `paper_arxiv.tex` (lines ~180-205,
  ~262-282).
- SWER metric definition: `compute_swer.py` + paper line 262.
- Training arg reference: `python train_model.py --help`.
- All cloud-training flags: README "Cloud Training" section.

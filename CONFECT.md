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
| `cloud_train.sh` | Remote destroy switched from raw `curl` to `vastai` CLI | The bare `curl -X DELETE` against Vast's REST API silently failed in our dry-runs, leaving zombie instances billing. The CLI uses the same API key file the local machine does, and is what successfully destroyed our orphan. The remote script now installs the `vastai` CLI before training and calls `vastai destroy instance` at the end. |

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

`cloud_train.sh` uses bash 4+ features. macOS ships bash 3.2 at
`/bin/bash`, which silently fails partway through. **Always invoke as
`./cloud_train.sh ...`** so the script's `#!/usr/bin/env bash` shebang
picks up Homebrew's bash (5+) from PATH. Don't prefix with `bash`.

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

## Swapping the trained model into the design-agent

After training finishes, the new checkpoint sits in `confect/google-font-classifier`.
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

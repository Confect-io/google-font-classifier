#!/usr/bin/env python3
"""
Create a tiny test dataset for dry-run testing of the cloud training pipeline.

If a local dataset exists at .data_out/font_dataset_v4/, samples 3 random
classes and copies a handful of real images. Otherwise, generates synthetic
224x224 placeholder images for 3 fake font classes.

Outputs:
  test_dataset_dry_run/
    train/<ClassName>/{0..9}.png    (10 per class)
    test/<ClassName>/{0..2}.png     (3 per class)
    train.tar                       (tar of train/)
    test.tar                        (tar of test/)

Usage:
  python create_test_dataset.py                  # auto-detect local data
  python create_test_dataset.py --upload          # also upload to HuggingFace
  python create_test_dataset.py --synthetic       # force synthetic images
"""

import argparse
import os
import random
import shutil
import tarfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_CLASSES = 3
TRAIN_PER_CLASS = 10
TEST_PER_CLASS = 3
IMG_HEIGHT = 224
IMG_WIDTH = 224  # placeholder width; real images vary
OUTPUT_DIR = Path(__file__).parent / "test_dataset_dry_run"
LOCAL_DATASET = Path(__file__).parent / ".data_out" / "font_dataset_v4"
HF_REPO = "dchen0/font_crops_test"

# Fake class names used when no local data is available
FAKE_CLASSES = ["FakeSerif_Regular", "FakeSans_Bold", "FakeMono_Light"]


def generate_synthetic_image(class_index: int, image_index: int) -> "Image":
    """Generate a simple synthetic 224x224 PNG distinguishable per class."""
    from PIL import Image, ImageDraw, ImageFont

    # Distinct background colours per class
    bg_colors = [(200, 220, 240), (240, 220, 200), (220, 240, 200)]
    text_color = (30, 30, 30)
    bg = bg_colors[class_index % len(bg_colors)]

    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), bg)
    draw = ImageDraw.Draw(img)

    label = FAKE_CLASSES[class_index]
    text = f"{label}\nimg {image_index}"

    # Use default font (always available)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    # Centre the text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (IMG_WIDTH - tw) // 2
    y = (IMG_HEIGHT - th) // 2
    draw.text((x, y), text, fill=text_color, font=font)

    return img


def copy_real_images(src_dir: Path, dst_dir: Path, count: int):
    """Copy up to `count` random PNG images from src_dir into dst_dir."""
    pngs = sorted(src_dir.glob("*.png"))
    selected = random.sample(pngs, min(count, len(pngs)))
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(selected):
        shutil.copy2(src, dst_dir / f"{i}.png")


def make_tar(source_dir: Path, tar_path: Path):
    """Create a tar archive of source_dir (using the dir basename as root)."""
    basename = source_dir.name  # "train" or "test"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(str(source_dir), arcname=basename)
    size_kb = tar_path.stat().st_size / 1024
    print(f"  {tar_path.name}: {size_kb:.0f} KB")


def build_dataset(use_synthetic: bool = False):
    """Build the test dataset directory and tar files."""
    have_local = LOCAL_DATASET.exists() and (LOCAL_DATASET / "train").is_dir()
    use_real = have_local and not use_synthetic

    if use_real:
        print(f"Using real images from {LOCAL_DATASET}")
        all_classes = sorted(
            d.name
            for d in (LOCAL_DATASET / "train").iterdir()
            if d.is_dir() and any(d.glob("*.png"))
        )
        selected_classes = random.sample(all_classes, min(NUM_CLASSES, len(all_classes)))
    else:
        print("No local dataset found (or --synthetic). Generating placeholder images.")
        selected_classes = FAKE_CLASSES[:NUM_CLASSES]

    print(f"Classes: {selected_classes}")

    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    train_root = OUTPUT_DIR / "train"
    test_root = OUTPUT_DIR / "test"

    for ci, cls_name in enumerate(selected_classes):
        train_cls = train_root / cls_name
        test_cls = test_root / cls_name
        train_cls.mkdir(parents=True, exist_ok=True)
        test_cls.mkdir(parents=True, exist_ok=True)

        if use_real:
            copy_real_images(LOCAL_DATASET / "train" / cls_name, train_cls, TRAIN_PER_CLASS)
            copy_real_images(LOCAL_DATASET / "test" / cls_name, test_cls, TEST_PER_CLASS)
        else:
            for i in range(TRAIN_PER_CLASS):
                img = generate_synthetic_image(ci, i)
                img.save(train_cls / f"{i}.png")
            for i in range(TEST_PER_CLASS):
                img = generate_synthetic_image(ci, i)
                img.save(test_cls / f"{i}.png")

    # Summary
    total_train = sum(len(list((train_root / c).glob("*.png"))) for c in selected_classes)
    total_test = sum(len(list((test_root / c).glob("*.png"))) for c in selected_classes)
    print(f"\nDataset created at: {OUTPUT_DIR}")
    print(f"  {len(selected_classes)} classes, {total_train} train images, {total_test} test images")

    # Create tar files
    print("\nCreating tar archives...")
    make_tar(train_root, OUTPUT_DIR / "train.tar")
    make_tar(test_root, OUTPUT_DIR / "test.tar")

    return selected_classes


def upload_to_hf():
    """Upload train.tar and test.tar to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)

    for fname in ["train.tar", "test.tar"]:
        fpath = OUTPUT_DIR / fname
        if not fpath.exists():
            print(f"  Skipping {fname} (not found)")
            continue
        print(f"  Uploading {fname} ({fpath.stat().st_size / 1024:.0f} KB)...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=HF_REPO,
            repo_type="dataset",
        )
    print(f"\nUploaded to: https://huggingface.co/datasets/{HF_REPO}")


def main():
    parser = argparse.ArgumentParser(description="Create a tiny test dataset for dry-run training")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic images even if local dataset exists")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace after building")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    build_dataset(use_synthetic=args.synthetic)

    if args.upload:
        print("\nUploading to HuggingFace...")
        upload_to_hf()


if __name__ == "__main__":
    main()

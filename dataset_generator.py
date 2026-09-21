#!/usr/bin/env python3
"""Render synthetic text crops for DINOv2 fine-tuning.

One class per FAMILY (a directory under --font_dir). Weight is sampled per
image from whatever .ttf files that directory holds, so the model learns a
family across weights instead of treating each weight as its own class — the
design agent discards predicted weight anyway.

Run:
    uv run --with numpy --with pillow --with tqdm python3 dataset_generator.py \\
        --font_dir ./fonts --out_dir ./data --train_per_class 1500
"""
import argparse
import logging
import multiprocessing
import os
import pathlib
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

_TEXT_CORPUS = None

# ---------------------------------------------------------------------------
# Augmentation ranges.
#
# These exist to close the gap between what we render and what the model
# actually sees: a JPEG-compressed, often low-resolution, usually SINGLE-LINE
# crop handed over by OCR. v5 trained on pristine multi-line PNG blocks, which
# is none of those things.
# ---------------------------------------------------------------------------
SINGLE_LINE_P = 0.75      # OCR emits text lines, so most samples are one line
TARGET_H = (22, 150)      # px per line after downsampling — real crops are small
JPEG_QUALITY = (55, 95)   # every production image has been through JPEG
NOISE_SIGMA = (0.0, 0.06) # fraction of 255; v5 used a fixed 0.10 on every image
CONTRAST_MIN = (60, 120)  # luminance gap between text and background


def _load_text_corpus():
    d = pathlib.Path("input_data")
    if not d.exists():
        raise ValueError(f"Input data directory {d} does not exist")
    texts = [f.read_text(encoding="utf-8").strip() for f in sorted(d.glob("*.txt"))]
    texts = [t for t in texts if len(t) >= 100]
    if not texts:
        raise ValueError(f"No usable text files in {d}")
    return texts


def choose_sentence(corpus, single_line: bool):
    content = random.choice(corpus)
    # Short strings for single-line crops, matching an OCR text box; longer
    # ones only when we deliberately build a multi-line block.
    n = random.randint(8, 38) if single_line else random.randint(40, 110)
    start = random.randint(0, max(0, len(content) - n))
    s = content[start:start + n]
    if not single_line:
        s = "".join("\n" if c == " " and random.random() < 0.18 else c for c in s)
    s = s.strip()
    if single_line:
        s = s.replace("\n", " ")
        # Ad copy is frequently set in caps; the corpus never is.
        if random.random() < 0.18:
            s = s.upper()
    return s or None


def _rand_rgb():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def _lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def render_and_crop(text, font, padding, img_size):
    bg = _rand_rgb()
    bg_lum = _lum(bg)
    floor = random.randint(*CONTRAST_MIN)
    fg = None
    for _ in range(40):
        cand = _rand_rgb()
        if abs(bg_lum - _lum(cand)) >= floor:
            fg = cand
            break
    if fg is None:
        fg = (255, 255, 255) if bg_lum < 128 else (0, 0, 0)

    lines = [ln for ln in text.split("\n") if ln.strip()] or [text]
    line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
    spacing = int(line_h * random.uniform(0.12, 0.4))

    # Wrap only multi-line blocks. A single line stays long and thin on
    # purpose — that is the shape OCR hands us.
    if len(lines) > 1:
        max_px = line_h * random.uniform(6, 12)
        wrapped = []
        for line in lines:
            words = line.split(" ")
            cur = words[0] if words else ""
            for w in words[1:]:
                test = cur + " " + w
                bb = font.getbbox(test)
                if bb[2] - bb[0] > max_px:
                    wrapped.append(cur)
                    cur = w
                else:
                    cur = test
            wrapped.append(cur)
        lines = wrapped

    total_h = len(lines) * line_h + (len(lines) - 1) * spacing
    max_w = max((font.getbbox(l)[2] - font.getbbox(l)[0]) for l in lines if l.strip())
    if max_w <= 0:
        return None

    cw, ch = int(max_w) + padding * 2, int(total_h) + padding * 2
    canvas = Image.new("RGB", (cw, ch), bg)
    draw = ImageDraw.Draw(canvas)
    align = random.choice(["left", "center", "right"])
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        bb = font.getbbox(line)
        lw = bb[2] - bb[0]
        x = padding if align == "left" else (cw - lw) // 2 if align == "center" else cw - lw - padding
        draw.text((x, padding + i * (line_h + spacing)), line, fill=fg, font=font, anchor="lt")

    bbox = canvas.getbbox()
    if not bbox:
        return None
    # OCR boxes are not perfect glyph bounds: they carry a little slack, and
    # they sometimes shave an ascender or a final letter. v5 cropped to the
    # exact bbox every time, so the model never saw either.
    x0, y0, x1, y1 = bbox
    pad_y = max(1, int((y1 - y0) * 0.10))
    pad_x = max(1, int((y1 - y0) * 0.18))
    x0 += random.randint(-pad_x, pad_x)
    x1 += random.randint(-pad_x, pad_x)
    y0 += random.randint(-pad_y, pad_y)
    y1 += random.randint(-pad_y, pad_y)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(canvas.width, x1), min(canvas.height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    glyph = canvas.crop((x0, y0, x1, y1))

    # Resolution round-trip. Real crops are 20-40px tall and get upscaled by
    # the preprocessing; v5 rendered everything pristine and large, so the
    # model never saw the softness that implies.
    per_line = random.randint(*TARGET_H)
    target_h = min(per_line * len(lines), img_size * 3)
    target_w = max(4, int(target_h * glyph.width / glyph.height))
    if target_w > img_size * 12:  # guard absurd aspect ratios
        return None
    glyph = glyph.resize((target_w, target_h), Image.Resampling.LANCZOS)

    sigma = random.uniform(*NOISE_SIGMA)
    if sigma > 0:
        arr = np.asarray(glyph, dtype=np.float32)
        arr += np.random.normal(0, sigma * 255, arr.shape).astype(np.float32)
        glyph = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return glyph


def _worker_init(corpus):
    global _TEXT_CORPUS
    _TEXT_CORPUS = corpus


def _generate_family(args):
    """All images for one family. Weight is sampled per image."""
    (family, ttf_paths, train_dir, test_dir, font_size, img_size, padding,
     n_train, n_test, no_clobber) = args

    fonts = []
    for p in ttf_paths:
        try:
            fonts.append(ImageFont.truetype(p, font_size,
                                            layout_engine=ImageFont.Layout.BASIC))
        except Exception as e:
            logger.warning("load failed %s: %s", p, e)
    if not fonts:
        return family, 0, 0

    corpus = _TEXT_CORPUS
    made = {"train": 0, "test": 0}

    def emit(root, idx):
        font = random.choice(fonts)
        single = random.random() < SINGLE_LINE_P
        roll = random.random()
        if roll < 0.10:
            text = f"{random.randint(1, 1000000)}"
        elif roll < 0.18:
            text = f"${random.randint(1, 100000)}"
        elif roll < 0.25:
            text = f"{random.randint(0, 100)}%"
        else:
            text = choose_sentence(corpus, single)
        if not text:
            return False
        img = render_and_crop(text, font, padding, img_size)
        if img is None:
            return False
        dest = root / f"{idx}.jpg"
        if no_clobber and dest.exists():
            return True
        img.save(dest, "JPEG", quality=random.randint(*JPEG_QUALITY),
                 optimize=False, subsampling=2)
        return True

    for split, count, base in (("train", n_train, pathlib.Path(train_dir)),
                               ("test", n_test, pathlib.Path(test_dir))):
        root = base / family
        root.mkdir(parents=True, exist_ok=True)
        i = 0
        attempts = 0
        while made[split] < count and attempts < count * 3:
            attempts += 1
            if emit(root, i):
                made[split] += 1
                i += 1
    return family, made["train"], made["test"]


def build_dataset(font_dir, out_dir, font_size, img_size, padding, no_clobber,
                  workers, n_train, n_test):
    font_dir, out_dir = pathlib.Path(font_dir), pathlib.Path(out_dir)
    train_dir, test_dir = out_dir / "train", out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # The directory IS the class. No allowlist to keep in sync any more.
    families = []
    for d in sorted(p for p in font_dir.iterdir() if p.is_dir()):
        ttfs = sorted(str(f) for f in d.glob("*.ttf"))
        if ttfs:
            families.append((d.name, ttfs))
    if not families:
        sys.exit(f"No family directories with .ttf files under {font_dir}")

    weights_total = sum(len(t) for _, t in families)
    print(f"{len(families)} families, {weights_total} weight files "
          f"({weights_total / len(families):.1f} per family)")
    print(f"Target: {n_train} train + {n_test} test per family "
          f"= {len(families) * (n_train + n_test):,} images")

    corpus = _load_text_corpus()
    work = [(fam, ttfs, str(train_dir), str(test_dir), font_size, img_size,
             padding, n_train, n_test, no_clobber) for fam, ttfs in families]

    results = []
    with multiprocessing.Pool(workers, initializer=_worker_init,
                              initargs=(corpus,)) as pool:
        for r in tqdm(pool.imap_unordered(_generate_family, work),
                      total=len(work), unit="family"):
            results.append(r)

    short = [(f, tr) for f, tr, _ in results if tr < n_train]
    total_train = sum(tr for _, tr, _ in results)
    total_test = sum(te for _, _, te in results)
    print(f"\n--- Summary ---")
    print(f"  Families:     {len(results)}")
    print(f"  Train images: {total_train:,}")
    print(f"  Test images:  {total_test:,}")
    if short:
        print(f"  WARNING: {len(short)} families under target: {short[:5]}")
    print("Done.")


def cli():
    ap = argparse.ArgumentParser(description="Render font crops for DINOv2")
    ap.add_argument("--font_dir", required=True, help="Dir of per-family subdirs")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--img_size", type=int, default=256,
                    help="Reference size for the resolution round-trip")
    ap.add_argument("--font_size", type=int, default=150,
                    help="Render size before downsampling")
    ap.add_argument("--padding", type=int, default=40)
    ap.add_argument("--train_per_class", type=int, default=1500)
    ap.add_argument("--test_per_class", type=int, default=150)
    ap.add_argument("--no-clobber", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    build_dataset(args.font_dir, args.out_dir, args.font_size, args.img_size,
                  args.padding, args.no_clobber, args.workers or os.cpu_count() or 1,
                  args.train_per_class, args.test_per_class)


if __name__ == "__main__":
    cli()

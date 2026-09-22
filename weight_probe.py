#!/usr/bin/env python3
"""Can font weight be read off a crop mathematically, without a model?

The classifier predicts a FAMILY; weight is folded in as augmentation. This
probes whether the weight can be recovered afterwards by measuring ink.

Method
------
Stroke width relative to a size normaliser is the physical signal, but its
absolute value is a property of the typeface, not just the weight: Playfair
Regular and Montserrat Regular sit at very different ratios, and serifs have
modulated strokes. A global threshold cannot work.

What makes it tractable is that we already know the family. So:

  1. offline, render each family x weight CLEANLY and record its ratio
  2. at test time, measure the crop and take the nearest reference
     *within the predicted family*

Stroke width is estimated as 2 x mean(distance transform) over the skeleton,
which is robust to the shape of the glyphs. The normaliser is the ink
bounding-box height (see --normaliser for the alternative).

Run:
    uv run --with numpy --with pillow --with scikit-image python3 weight_probe.py
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageFont

from dataset_generator import _load_text_corpus, choose_sentence, render_and_crop

WEIGHT_ORDER = ["Light", "Regular", "Medium", "SemiBold", "Bold"]


def ink_mask(img: Image.Image) -> np.ndarray:
    """Binary mask of the text. Otsu on greyscale, then orient so the ink is
    the minority class — text is nearly always a smaller area than its
    background in a crop."""
    from skimage.filters import threshold_otsu

    g = np.asarray(img.convert("L"), dtype=np.float32)
    if g.max() - g.min() < 8:            # flat image, nothing to measure
        return np.zeros_like(g, dtype=bool)
    t = threshold_otsu(g)
    m = g < t
    if m.mean() > 0.5:
        m = ~m
    return m


NORMALISER = "bbox"


def core_band_height(m: np.ndarray) -> float:
    """Height of the densest horizontal band of ink — an x-height proxy.

    The ink bounding box is a poor size reference because it depends on which
    glyphs happen to be present: "ABC", "gyp" and "xyz" have very different
    heights at identical font size, so stroke/bbox moves with the TEXT rather
    than the weight. The rows carrying most of the ink approximate the
    x-height band (or cap band in all-caps), which is far more stable.
    """
    prof = m.sum(axis=1).astype(np.float32)
    if prof.max() <= 0:
        return 0.0
    rows = np.where(prof >= 0.5 * prof.max())[0]
    return float(rows[-1] - rows[0] + 1) if len(rows) else 0.0


def stroke_and_height(img: Image.Image) -> tuple[float, float] | None:
    """(stroke width, size reference) in pixels, or None if unmeasurable."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    m = ink_mask(img)
    if m.sum() < 30:
        return None
    dt = distance_transform_edt(m)
    sk = skeletonize(m)
    if sk.sum() < 5:
        return None
    stroke = 2.0 * float(dt[sk].mean())
    if NORMALISER == "core":
        h = core_band_height(m)
    else:
        rows = np.where(m.any(axis=1))[0]
        h = float(rows[-1] - rows[0] + 1) if len(rows) >= 2 else 0.0
    return (stroke, h) if h > 1 else None


def ratio_of(img: Image.Image) -> float | None:
    r = stroke_and_height(img)
    if r is None:
        return None
    stroke, height = r
    return stroke / height if height > 0 else None


def clean_render(ttf: str, text: str, px: int = 220) -> Image.Image:
    """Reference render: no colour, noise, downsampling or JPEG."""
    from PIL import ImageDraw

    font = ImageFont.truetype(ttf, px, layout_engine=ImageFont.Layout.BASIC)
    bb = font.getbbox(text)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    pad = px // 2
    c = Image.new("RGB", (w + pad * 2, h + pad * 2), (255, 255, 255))
    ImageDraw.Draw(c).text((pad, pad), text, fill=(0, 0, 0), font=font, anchor="lt")
    return c.crop(c.getbbox() or (0, 0, c.width, c.height))


def jpeg(img: Image.Image, q: int) -> Image.Image:
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--families", type=int, default=5)
    ap.add_argument("--per_weight", type=int, default=4,
                    help="test crops per (family, weight)")
    ap.add_argument("--fixed_text", action="store_true",
                    help="use one string everywhere — the ideal case, which "
                         "isolates weight from text-shape effects")
    ap.add_argument("--font_dir", default="./fonts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrated_refs", action="store_true",
                    help="build references from DEGRADED renders (the same "
                         "downsample + noise + JPEG as the test crops) rather "
                         "than clean ones. Clean references sit systematically "
                         "too thin, because low-res antialiasing plus Otsu "
                         "fattens ink, so every error lands one weight heavy.")
    ap.add_argument("--ref_samples", type=int, default=8)
    ap.add_argument("--normaliser", choices=["bbox", "core"], default="bbox")
    args = ap.parse_args()
    global NORMALISER
    NORMALISER = args.normaliser
    random.seed(args.seed)
    np.random.seed(args.seed)

    full = [d for d in sorted(os.listdir(args.font_dir))
            if os.path.isdir(f"{args.font_dir}/{d}")
            and len(glob.glob(f"{args.font_dir}/{d}/*.ttf")) == 5]
    fams = random.sample(full, min(args.families, len(full)))
    print(f"{len(full)} families carry all 5 weights; probing {len(fams)}: "
          f"{', '.join(fams)}\n")

    REF_TEXT = "Hamburgefonstiv"
    corpus0 = _load_text_corpus()
    refs: dict[tuple[str, str], float] = {}
    for fam in fams:
        for w in WEIGHT_ORDER:
            ttf = f"{args.font_dir}/{fam}/{fam}-{w}.ttf"
            if not os.path.exists(ttf):
                continue
            if args.calibrated_refs:
                font = ImageFont.truetype(ttf, 150, layout_engine=ImageFont.Layout.BASIC)
                vals = []
                for _ in range(args.ref_samples * 4):
                    if len(vals) >= args.ref_samples:
                        break
                    img = render_and_crop(REF_TEXT, font, 40, 256)
                    if img is None:
                        continue
                    v = ratio_of(jpeg(img, random.randint(55, 95)))
                    if v is not None:
                        vals.append(v)
                if vals:
                    refs[(fam, w)] = float(np.median(vals))
            else:
                r = ratio_of(clean_render(ttf, REF_TEXT))
                if r is not None:
                    refs[(fam, w)] = r

    print("reference stroke/height ratios (clean renders):")
    print(f"  {'family':<22}" + "".join(f"{w:>10}" for w in WEIGHT_ORDER))
    for fam in fams:
        row = "".join(f"{refs.get((fam, w), float('nan')):>10.4f}" for w in WEIGHT_ORDER)
        print(f"  {fam:<22}{row}")

    mono = sum(
        all(refs[(f, a)] < refs[(f, b)]
            for a, b in zip(WEIGHT_ORDER, WEIGHT_ORDER[1:])
            if (f, a) in refs and (f, b) in refs)
        for f in fams
    )
    print(f"\n  strictly increasing with weight in {mono}/{len(fams)} families")

    corpus = _load_text_corpus()
    total = hits = near = unmeasurable = 0
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    per_weight = Counter()
    per_weight_hit = Counter()

    for fam in fams:
        for w in WEIGHT_ORDER:
            ttf = f"{args.font_dir}/{fam}/{fam}-{w}.ttf"
            if (fam, w) not in refs:
                continue
            font = ImageFont.truetype(ttf, 150, layout_engine=ImageFont.Layout.BASIC)
            made = 0
            attempts = 0
            while made < args.per_weight and attempts < args.per_weight * 8:
                attempts += 1
                text = REF_TEXT if args.fixed_text else choose_sentence(corpus, True)
                if not text:
                    continue
                img = render_and_crop(text, font, 40, 256)
                if img is None:
                    continue
                img = jpeg(img, random.randint(55, 95))
                made += 1
                total += 1
                r = ratio_of(img)
                if r is None:
                    unmeasurable += 1
                    continue
                pred = min(refs, key=lambda k: abs(refs[k] - r) if k[0] == fam else 1e9)[1]
                confusion[(w, pred)] += 1
                per_weight[w] += 1
                if pred == w:
                    hits += 1
                    per_weight_hit[w] += 1
                if abs(WEIGHT_ORDER.index(pred) - WEIGHT_ORDER.index(w)) <= 1:
                    near += 1

    # What you would actually ship. Adjacent weights are below the
    # resolution limit of a 20-40px crop, but the coarse buckets a designer
    # cares about are not.
    BUCKET = {"Light": "Light", "Regular": "Regular", "Medium": "Regular",
              "SemiBold": "Bold", "Bold": "Bold"}
    bucket_hits = sum(n for (t, p_), n in confusion.items() if BUCKET[t] == BUCKET[p_])
    bucket_total = sum(confusion.values())

    print(f"\n--- {total} test crops "
          f"({'fixed text' if args.fixed_text else 'varied text'}) ---")
    print(f"exact weight:        {hits}/{total} = {hits/total:.1%}   (chance 20%)")
    print(f"within one step:     {near}/{total} = {near/total:.1%}")
    print(f"unmeasurable:        {unmeasurable}")
    if bucket_total:
        print(f"3-bucket (Light/Regular/Bold): {bucket_hits}/{bucket_total} = "
              f"{bucket_hits/bucket_total:.1%}   (chance 33%)")

    print(f"\nconfusion (row = truth, col = predicted):")
    print(f"  {'':<10}" + "".join(f"{w:>10}" for w in WEIGHT_ORDER))
    for t in WEIGHT_ORDER:
        row = "".join(f"{confusion.get((t, p), 0):>10}" for p in WEIGHT_ORDER)
        acc = per_weight_hit[t] / per_weight[t] if per_weight[t] else 0
        print(f"  {t:<10}{row}   {acc:>6.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

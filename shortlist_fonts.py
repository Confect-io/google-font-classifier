#!/usr/bin/env python3
"""Build the human-review shortlist for font curation (RETRAIN_PLAN.md step 1a).

Narrows the Google Fonts catalogue to a reviewable candidate set, embeds each
family with the SAME DINOv2 backbone the classifier fine-tunes, and clusters
by visual similarity. The clusters are NOT picks -- they are the review
layout, so that a screen full of near-identical grotesks reads as "these are
interchangeable to the model, keep two" instead of scrolling past as fourteen
separate entries.

Output feeds the picker web page. A human makes every include/exclude call.

Run (uv handles the deps ephemerally):
    uv run --with torch --with transformers --with scikit-learn \\
        --with pillow --with fonttools --with numpy \\
        python3 shortlist_fonts.py \\
            --google_fonts_repo ../google_fonts_repo \\
            --per_category 250 \\
            --out ./font_candidates.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("fontTools").setLevel(logging.ERROR)

_METADATA_URL = "https://fonts.google.com/metadata/fonts"
_LICENSE_DIRS = ("ofl", "apache", "ufl")

# The backbone the classifier fine-tunes. Embedding candidates with the same
# encoder means "visually similar" is measured in the space the model actually
# learns in, not a proxy.
_ENCODER = "facebook/dinov2-base"
_INPUT_SIZE = 224

# Classic type-specimen strings: enough of the alphabet, both cases, and the
# numerals/currency that ad copy is full of.
_SPECIMENS = (
    "Hamburgefonstiv",
    "ABCDEFGHIJK",
    "abcdefghijk",
    "0123456789 $49 25%",
)

# Google's `category` is coarse and `stroke` disambiguates the Display bucket
# (a Display family can be sans, serif or neither). Collapse both into the
# five values the design-agent's font_categories.json already accepts.
_DESIGN_CATEGORIES = ("sans-serif", "serif", "display", "handwritten", "monospace")


def _design_category(fam: dict) -> str:
    category = fam.get("category", "")
    if category == "Handwriting":
        return "handwritten"
    if category == "Monospace":
        return "monospace"
    if category == "Sans Serif":
        return "sans-serif"
    if category == "Serif":
        return "serif"
    if category == "Display":
        return "display"
    return "display"


def fetch_metadata(cache: Path | None) -> dict:
    if cache and cache.is_file():
        log.info("Using cached metadata: %s", cache)
        return json.loads(cache.read_text())
    log.info("Fetching Google Fonts metadata ...")
    with urllib.request.urlopen(_METADATA_URL) as resp:
        meta = json.load(resp)
    if cache:
        cache.write_text(json.dumps(meta))
    return meta


def family_to_slug(family: str) -> str:
    return family.lower().replace(" ", "")


def family_stem(family: str) -> str:
    """Canonical stem used as the class/directory name downstream."""
    return re.sub(r"[^A-Za-z0-9]", "", family)


def find_family_dir(repo: Path, slug: str) -> Path | None:
    for lic in _LICENSE_DIRS:
        d = repo / lic / slug
        if d.is_dir():
            return d
    return None


def upright_ttf(family_dir: Path) -> Path | None:
    """The best upright .ttf to render a specimen from: prefer a variable font
    (we pin it to Regular), else a static Regular, else any non-italic."""
    statics: list[Path] = []
    for ttf in sorted(family_dir.rglob("*.ttf")):
        stem = ttf.stem
        head = stem.split("[", 1)[0]
        if head.endswith("-Italic") or "Italic" in stem:
            continue
        if "[" in stem:
            return ttf
        statics.append(ttf)
    for ttf in statics:
        if ttf.stem.endswith("-Regular"):
            return ttf
    return statics[0] if statics else None


def render_specimen(ttf: Path, text: str, font_size: int = 180):
    """Black-on-white, tight-cropped, padded to square, resized to the encoder
    input. No colour or noise augmentation -- we are measuring typeface shape,
    not robustness."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(ttf), font_size, layout_engine=ImageFont.Layout.BASIC)
    try:
        font.set_variation_by_name("Regular")
    except Exception:
        pass  # static font, or no instance named Regular -- default is fine

    bbox = font.getbbox(text)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return None

    pad = font_size // 2
    canvas = Image.new("RGB", (w + pad * 2, h + pad * 2), (255, 255, 255))
    ImageDraw.Draw(canvas).text((pad, pad), text, fill=(0, 0, 0), font=font, anchor="lt")

    inverted = Image.eval(canvas, lambda p: 255 - p)
    crop = inverted.getbbox()
    if not crop:
        return None
    glyph = canvas.crop(crop)

    side = max(glyph.size)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(glyph, ((side - glyph.width) // 2, (side - glyph.height) // 2))
    return square.resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.LANCZOS)


def embed_families(candidates: list[dict], repo: Path, batch_size: int = 32):
    """Mean of the per-specimen [CLS] embeddings, L2-normalised. Returns the
    candidates that rendered successfully, aligned with the embedding matrix."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    log.info("Loading %s on %s ...", _ENCODER, device)
    processor = AutoImageProcessor.from_pretrained(_ENCODER)
    model = AutoModel.from_pretrained(_ENCODER).to(device).eval()

    images: list[Image.Image] = []
    owners: list[int] = []  # index into `kept`, one entry per rendered specimen
    kept: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for cand in candidates:
        fdir = find_family_dir(repo, cand["slug"])
        if fdir is None:
            skipped.append((cand["family"], "no directory in google/fonts clone"))
            continue
        ttf = upright_ttf(fdir)
        if ttf is None:
            skipped.append((cand["family"], "no upright .ttf"))
            continue
        rendered = []
        for text in _SPECIMENS:
            try:
                img = render_specimen(ttf, text)
            except Exception as exc:
                log.debug("  render failed %s: %s", cand["family"], exc)
                img = None
            if img is not None:
                rendered.append(img)
        if not rendered:
            skipped.append((cand["family"], "all specimens failed to render"))
            continue
        idx = len(kept)
        kept.append(cand)
        images.extend(rendered)
        owners.extend([idx] * len(rendered))

    log.info("Rendered %d specimens for %d families", len(images), len(kept))
    if skipped:
        log.info("Skipped %d families (e.g. %s)", len(skipped), skipped[:3])

    vectors = np.zeros((len(images), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            cls = model(**inputs).last_hidden_state[:, 0]
            vectors[start : start + len(batch)] = cls.float().cpu().numpy()
            if start % (batch_size * 10) == 0:
                log.info("  embedded %d/%d", start, len(images))

    embeddings = np.zeros((len(kept), vectors.shape[1]), dtype=np.float32)
    owners_arr = np.array(owners)
    for i in range(len(kept)):
        embeddings[i] = vectors[owners_arr == i].mean(axis=0)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    return kept, embeddings, skipped


def cluster_within_categories(
    kept: list[dict], embeddings: np.ndarray, quantile: float,
    absolute: float | None = None,
) -> dict[str, float]:
    """Agglomerative clustering per design category. Writes `cluster`,
    `nearest` and `nearest_distance` onto each candidate in place.

    Clustering inside a category (not across) keeps the review layout honest:
    a script face and a grotesk are never "the same cluster" just because
    they are both far from everything else.

    The linkage threshold is a QUANTILE of each category's own pairwise
    distance distribution, not a fixed number. Categories have wildly
    different intrinsic spreads -- every monospace face sits within ~0.006 of
    every other, while display faces range past 0.12 -- so one absolute
    threshold would collapse mono into a single blob and shatter display into
    singletons. Pass `absolute` to override with a fixed value.
    """
    from sklearn.cluster import AgglomerativeClustering

    by_category: dict[str, list[int]] = defaultdict(list)
    for i, cand in enumerate(kept):
        by_category[cand["category"]].append(i)

    thresholds: dict[str, float] = {}
    next_cluster_id = 0
    for category, idxs in by_category.items():
        sub = embeddings[idxs]
        if len(idxs) == 1:
            kept[idxs[0]]["cluster"] = next_cluster_id
            next_cluster_id += 1
            thresholds[category] = 0.0
        else:
            dists = 1.0 - (sub @ sub.T)
            iu = np.triu_indices(len(idxs), k=1)
            threshold = absolute if absolute is not None else float(
                np.quantile(dists[iu], quantile)
            )
            thresholds[category] = round(threshold, 5)
            labels = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric="cosine",
                linkage="average",
            ).fit_predict(sub)
            remap: dict[int, int] = {}
            for local, label in enumerate(labels):
                if label not in remap:
                    remap[label] = next_cluster_id
                    next_cluster_id += 1
                kept[idxs[local]]["cluster"] = remap[label]

        # Nearest neighbour within the category, for the "how redundant is
        # this pick" hint on each card.
        sims = sub @ sub.T
        np.fill_diagonal(sims, -np.inf)
        for local, i in enumerate(idxs):
            if len(idxs) == 1:
                kept[i]["nearest"] = None
                kept[i]["nearest_distance"] = None
                continue
            j = int(np.argmax(sims[local]))
            kept[i]["nearest"] = kept[idxs[j]]["family"]
            kept[i]["nearest_distance"] = round(float(1.0 - sims[local][j]), 4)
        log.info("  %-12s %3d families -> %3d clusters (threshold %.4f)",
                 category, len(idxs),
                 len({kept[i]["cluster"] for i in idxs}), thresholds[category])
    return thresholds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--google_fonts_repo", type=Path, required=True)
    ap.add_argument("--per_category", type=int, default=250,
                    help="Keep the top N by popularity in each design category")
    ap.add_argument("--cluster_quantile", type=float, default=0.10,
                    help="Linkage threshold as a quantile of each category's own "
                         "pairwise distance distribution (default 0.10)")
    ap.add_argument("--cluster_threshold", type=float, default=None,
                    help="Absolute cosine-distance threshold, overriding "
                         "--cluster_quantile for every category")
    ap.add_argument("--out", type=Path, default=Path("./font_candidates.json"))
    ap.add_argument("--embeddings_out", type=Path, default=Path("./font_embeddings.npz"))
    ap.add_argument("--metadata_cache", type=Path, default=Path("./.gf_metadata.json"))
    args = ap.parse_args()

    if not args.google_fonts_repo.is_dir():
        sys.exit(f"google_fonts_repo not found: {args.google_fonts_repo}")

    meta = fetch_metadata(args.metadata_cache)
    families = meta["familyMetadataList"]
    log.info("Catalogue: %d families", len(families))

    # Latin-primary only. `primaryScript` is empty or "Latn" for Latin-designed
    # fonts; a font whose primary intent is Arabic/Devanagari/CJK often ships a
    # Latin subset as fallback, but those glyphs are secondary and should not
    # be trained as Latin classes.
    latin = [
        f for f in families
        if "latin" in f.get("subsets", [])
        and f.get("primaryScript", "") in ("", "Latn")
    ]
    log.info("Latin-primary: %d families", len(latin))

    by_category: dict[str, list[dict]] = defaultdict(list)
    for fam in latin:
        by_category[_design_category(fam)].append(fam)

    candidates: list[dict] = []
    for category in _DESIGN_CATEGORIES:
        pool = sorted(by_category.get(category, []), key=lambda f: f.get("popularity", 10**9))
        head = pool[: args.per_category]
        log.info("  %-12s %3d available -> %3d shortlisted", category, len(pool), len(head))
        for fam in head:
            candidates.append({
                "family": fam["family"],
                "stem": family_stem(fam["family"]),
                "slug": family_to_slug(fam["family"]),
                "category": category,
                "gf_category": fam.get("category", ""),
                "stroke": fam.get("stroke") or None,
                "popularity": fam.get("popularity"),
                "designers": fam.get("designers", [])[:2],
                "variable": bool(fam.get("axes")),
            })
    log.info("Shortlist: %d families to embed", len(candidates))

    kept, embeddings, skipped = embed_families(candidates, args.google_fonts_repo)

    log.info("Clustering (quantile %.2f of each category's distances) ...",
             args.cluster_quantile)
    thresholds = cluster_within_categories(
        kept, embeddings, args.cluster_quantile, args.cluster_threshold
    )

    # Biggest clusters first: those are the redundant crowds that most need a
    # human to thin them out.
    sizes = defaultdict(int)
    for cand in kept:
        sizes[cand["cluster"]] += 1
    for cand in kept:
        cand["cluster_size"] = sizes[cand["cluster"]]

    np.savez_compressed(
        args.embeddings_out,
        embeddings=embeddings,
        families=np.array([c["family"] for c in kept]),
    )
    payload = {
        "generated_from": _ENCODER,
        "specimens": list(_SPECIMENS),
        "cluster_quantile": args.cluster_quantile,
        "cluster_thresholds": thresholds,
        "per_category": args.per_category,
        "counts": {c: sum(1 for k in kept if k["category"] == c) for c in _DESIGN_CATEGORIES},
        "clusters": len(sizes),
        "skipped": [{"family": f, "reason": r} for f, r in skipped],
        "families": kept,
    }
    args.out.write_text(json.dumps(payload, indent=1))
    log.info("Wrote %d candidates in %d clusters to %s", len(kept), len(sizes), args.out)
    log.info("Wrote embeddings to %s", args.embeddings_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

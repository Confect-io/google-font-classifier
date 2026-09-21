#!/usr/bin/env python3
"""Materialise the chosen font vocabulary as static .ttf files.

This script no longer SELECTS anything — selection happens in the picker and
`fill_vocabulary.py`. It takes the finished family list and writes, per
family, one static .ttf per weight:

    fonts/
      Montserrat/
        Montserrat-Light.ttf
        Montserrat-Regular.ttf
        ...
      Lora/
        ...

The directory is the class. dataset_generator.py samples a weight at random
per image, so weight becomes augmentation WITHIN a family rather than a class
of its own — the design agent discards predicted weight anyway.

Run:
    uv run --with fonttools python3 curate_fonts.py \\
        --vocabulary ./vocabulary_proposal.json \\
        --google_fonts_repo ../google_fonts_repo \\
        --weights 300 400 500 600 700 \\
        --out_dir ./fonts
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

try:
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
except ImportError:
    sys.exit("fontTools required: uv run --with fonttools python3 curate_fonts.py ...")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("fontTools").setLevel(logging.ERROR)

_LICENSE_DIRS = ("ofl", "apache", "ufl")
_WEIGHT_NAMES = {
    "100": "Thin", "200": "ExtraLight", "300": "Light", "400": "Regular",
    "500": "Medium", "600": "SemiBold", "700": "Bold", "800": "ExtraBold",
    "900": "Black",
}


def family_stem(family: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", family)


def find_family_dir(repo: Path, family: str) -> Path | None:
    slug = family.lower().replace(" ", "")
    for lic in _LICENSE_DIRS:
        d = repo / lic / slug
        if d.is_dir():
            return d
    return None


def variable_upright(family_dir: Path) -> Path | None:
    for ttf in sorted(family_dir.glob("*.ttf")):
        if "[" not in ttf.stem:
            continue
        if ttf.stem.split("[", 1)[0].endswith("-Italic"):
            continue
        return ttf
    return None


def instance_weight(src: Path, weight: int, dest: Path) -> bool:
    """Pin a variable font to one weight, leaving other axes at default."""
    try:
        font = TTFont(str(src))
    except Exception as exc:
        log.warning("    open failed %s: %s", src.name, exc)
        return False
    fvar = font.get("fvar")
    if fvar is None:
        return False
    axes = {a.axisTag: a for a in fvar.axes}
    wght = axes.get("wght")
    if wght is None or not (wght.minValue <= weight <= wght.maxValue):
        return False
    pinned = {"wght": float(weight)}
    for tag, axis in axes.items():
        if tag != "wght":
            pinned[tag] = float(axis.defaultValue)
    try:
        instantiateVariableFont(font, pinned).save(str(dest))
    except Exception as exc:
        log.warning("    instancing failed %s @ %d: %s", src.name, weight, exc)
        return False
    return True


def copy_statics(family_dir: Path, weight_names: list[str], out: Path,
                 stem: str) -> int:
    """Fall back to whatever upright static weights the family ships."""
    for base in (family_dir / "static", family_dir):
        if not base.is_dir():
            continue
        found = 0
        for ttf in sorted(base.glob("*.ttf")):
            name = ttf.stem
            if "[" in name or "Italic" in name or "-" not in name:
                continue
            w = name.rsplit("-", 1)[1]
            if w in weight_names:
                shutil.copy(ttf, out / f"{stem}-{w}.ttf")
                found += 1
        if found:
            return found
    # Nothing matched the requested weights — take the single closest thing
    # rather than dropping the family entirely.
    for base in (family_dir / "static", family_dir):
        if not base.is_dir():
            continue
        for ttf in sorted(base.glob("*.ttf")):
            if "[" in ttf.stem or "Italic" in ttf.stem:
                continue
            shutil.copy(ttf, out / f"{stem}-Regular.ttf")
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vocabulary", type=Path, default=Path("./vocabulary_proposal.json"))
    ap.add_argument("--google_fonts_repo", type=Path, required=True)
    ap.add_argument("--weights", nargs="+", default=["300", "400", "500", "600", "700"])
    ap.add_argument("--out_dir", type=Path, default=Path("./fonts"))
    args = ap.parse_args()

    for w in args.weights:
        if w not in _WEIGHT_NAMES:
            sys.exit(f"Unknown weight {w!r}; valid: {sorted(_WEIGHT_NAMES)}")
    weights = [(int(w), _WEIGHT_NAMES[w]) for w in args.weights]
    weight_names = [n for _, n in weights]

    vocab = json.loads(args.vocabulary.read_text())
    families = vocab["families"] if isinstance(vocab, dict) else vocab
    log.info("Vocabulary: %d families", len(families))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_files = 0
    per_family: list[tuple[str, int]] = []
    missing: list[str] = []

    for family in families:
        fdir = find_family_dir(args.google_fonts_repo, family)
        if fdir is None:
            missing.append(family)
            continue
        stem = family_stem(family)
        out = args.out_dir / stem
        out.mkdir(parents=True, exist_ok=True)

        var = variable_upright(fdir)
        made = 0
        if var is not None:
            for weight, name in weights:
                if instance_weight(var, weight, out / f"{stem}-{name}.ttf"):
                    made += 1
        if made == 0:
            made = copy_statics(fdir, weight_names, out, stem)

        if made == 0:
            missing.append(family)
            out.rmdir()
            continue
        total_files += made
        per_family.append((family, made))

    log.info("Wrote %d .ttf files across %d family dirs -> %s",
             total_files, len(per_family), args.out_dir)
    thin = [f for f, n in per_family if n < 2]
    if thin:
        log.info("Only one weight available for %d families: %s",
                 len(thin), ", ".join(thin[:12]))
    if missing:
        log.warning("MISSING %d families (not in the clone): %s",
                    len(missing), ", ".join(missing))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

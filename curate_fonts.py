#!/usr/bin/env python3
"""Curate fonts for design-agent classifier training.

Picks the top N popular Google Fonts (Latin Sans/Serif/Display only) and
extracts specific weight instances from the variable-font sources in a
local clone of github.com/google/fonts. Writes static .ttf files into
--out_dir and prints the FONT_ALLOWLIST literal ready to paste into
dataset_generator.py.

Popular Google Fonts ship as variable fonts only (no `static/` subfolder);
this script uses fontTools to instance them to fixed weights at curation
time, so the training pipeline only ever sees static .ttf files.

Setup (one-time):
    cd ~/Confect/misc
    git clone --depth 1 https://github.com/google/fonts.git google_fonts_repo

Run (uv handles the fontTools dep ephemerally — no install needed):
    uv run --with fonttools python3 curate_fonts.py \\
        --google_fonts_repo ../google_fonts_repo \\
        --top_n 150 \\
        --weights 400 700 \\
        --out_dir ./fonts \\
        --allowlist_out ./FONT_ALLOWLIST.py

Then paste the allowlist into dataset_generator.py:24.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import urllib.request
from pathlib import Path

try:
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
except ImportError:
    sys.exit(
        "fontTools is required. Run with:\n"
        "    uv run --with fonttools python3 curate_fonts.py ...\n"
        "or install with: pip install fontTools"
    )

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
# fontTools is chatty about each instancing step ("Restricted limits:",
# "Instantiating glyf/gvar tables", ...). Quiet it.
logging.getLogger("fontTools").setLevel(logging.WARNING)

_METADATA_URL = "https://fonts.google.com/metadata/fonts"
_CATEGORY_ALLOWED = {"Sans Serif", "Serif", "Display"}
_LICENSE_DIRS = ("ofl", "apache", "ufl")

# CSS weight code -> name used in the output filename. Matches the
# google/fonts convention so dataset_generator.py's
# `stem.split("-")[0]` filter recovers the family name cleanly.
_WEIGHT_NAMES = {
    "100": "Thin",
    "200": "ExtraLight",
    "300": "Light",
    "400": "Regular",
    "500": "Medium",
    "600": "SemiBold",
    "700": "Bold",
    "800": "ExtraBold",
    "900": "Black",
}


def fetch_metadata() -> dict:
    with urllib.request.urlopen(_METADATA_URL) as resp:
        return json.load(resp)


def family_to_slug(family: str) -> str:
    return family.lower().replace(" ", "")


def family_canonical_stem(family_display_name: str) -> str:
    """Output-filename stem we use for both the .ttf file and the
    FONT_ALLOWLIST entry. Strips non-alphanumerics from the canonical
    family name so "Open Sans" -> "OpenSans" and "PT Sans Narrow" ->
    "PTSansNarrow". Keeping it consistent across the variable-font and
    static-font branches means dataset_generator's filter
    (stem.split("[")[0].split("-")[0]) always recovers the same key."""
    return re.sub(r"[^A-Za-z0-9]", "", family_display_name)


def find_family_dir(repo: Path, slug: str) -> Path | None:
    for lic in _LICENSE_DIRS:
        d = repo / lic / slug
        if d.is_dir():
            return d
    return None


def _variable_upright_ttf(family_dir: Path) -> Path | None:
    """Return the upright variable .ttf for a family, if one exists.

    Variable fonts have a filename like `Roboto[wdth,wght].ttf` (axes
    listed in square brackets). Italic variants follow the
    `Roboto-Italic[wght].ttf` naming; we skip those.
    """
    for ttf in sorted(family_dir.glob("*.ttf")):
        stem = ttf.stem
        if "[" not in stem:
            continue
        head = stem.split("[", 1)[0]
        if head.endswith("-Italic"):
            continue
        return ttf
    return None


def _instance_variable(
    src: Path, family_stem: str, weight: int, weight_name: str, out_dir: Path
) -> Path | None:
    """Extract a single weight instance from a variable font and write it
    out as a static .ttf. Other variation axes (wdth, opsz, ...) are
    pinned to their default values."""
    try:
        font = TTFont(str(src))
    except Exception as exc:
        log.warning("  open failed for %s: %s", src.name, exc)
        return None

    fvar = font.get("fvar")
    if fvar is None:
        log.warning("  no fvar axes in %s; not a variable font", src.name)
        return None

    axes = {axis.axisTag: axis for axis in fvar.axes}
    wght_axis = axes.get("wght")
    if wght_axis is None:
        log.warning("  %s has no wght axis; skipping", src.name)
        return None

    if not (wght_axis.minValue <= weight <= wght_axis.maxValue):
        log.info(
            "  %s weight %d outside axis range [%g, %g]; skipping",
            src.name, weight, wght_axis.minValue, wght_axis.maxValue,
        )
        return None

    pinned = {"wght": float(weight)}
    for tag, axis in axes.items():
        if tag != "wght":
            pinned[tag] = float(axis.defaultValue)

    try:
        static = instantiateVariableFont(font, pinned)
    except Exception as exc:
        log.warning("  instancing failed for %s @ %d: %s", src.name, weight, exc)
        return None

    out_path = out_dir / f"{family_stem}-{weight_name}.ttf"
    static.save(str(out_path))
    return out_path


def _static_matches(
    family_dir: Path, weight_names: list[str]
) -> list[tuple[Path, str]]:
    """Find non-italic static .ttf files matching the chosen weight names.

    Returns (source_path, weight_name) pairs so the caller can rename on
    copy. Looks first in `static/` then in the family root. The weight
    name is recovered from the LAST hyphen segment, ignoring "Web" /
    "Narrow" / "Caption" subfamily markers in the middle.
    """
    for base in (family_dir / "static", family_dir):
        if not base.is_dir():
            continue
        matches: list[tuple[Path, str]] = []
        for ttf in sorted(base.glob("*.ttf")):
            stem = ttf.stem
            if "[" in stem:
                continue
            if "Italic" in stem:
                continue
            if "-" not in stem:
                continue
            weight_part = stem.rsplit("-", 1)[1]
            if weight_part in weight_names:
                matches.append((ttf, weight_part))
        if matches:
            return matches
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--google_fonts_repo",
        type=Path,
        required=True,
        help="Local clone of github.com/google/fonts",
    )
    parser.add_argument("--top_n", type=int, default=150)
    parser.add_argument(
        "--weights",
        nargs="+",
        default=["400", "700"],
        help="CSS weight codes to include (default: 400 700)",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("./fonts"))
    parser.add_argument(
        "--allowlist_out",
        type=Path,
        default=None,
        help="Write the Python allowlist literal here (default: stdout)",
    )
    args = parser.parse_args()

    if not args.google_fonts_repo.is_dir():
        sys.exit(
            f"google_fonts_repo not found: {args.google_fonts_repo}\n"
            f"Clone it with:\n"
            f"    git clone --depth 1 https://github.com/google/fonts.git "
            f"{args.google_fonts_repo}"
        )

    for w in args.weights:
        if w not in _WEIGHT_NAMES:
            sys.exit(f"Unknown weight code {w!r}; valid: {sorted(_WEIGHT_NAMES)}")
    weight_pairs = [(int(w), _WEIGHT_NAMES[w]) for w in args.weights]
    weight_names = [name for _, name in weight_pairs]

    log.info("Fetching Google Fonts metadata ...")
    meta = fetch_metadata()
    families = meta["familyMetadataList"]
    log.info("Total families in catalogue: %d", len(families))

    families.sort(key=lambda f: f.get("popularity", 10**9))

    selected: list[dict] = []
    for fam in families:
        if len(selected) >= args.top_n:
            break
        if fam.get("category", "") not in _CATEGORY_ALLOWED:
            continue
        if "latin" not in fam.get("subsets", []):
            continue
        # primaryScript is empty (or "Latn") for Latin-designed fonts and
        # set to "Arab"/"Deva"/"Thai"/"Jpan"/"Kore"/etc. for fonts whose
        # primary intent is a non-Latin script. They often include a
        # Latin subset for fallback, but their Latin glyphs are
        # secondary and we shouldn't train them as Latin classes.
        primary = fam.get("primaryScript", "")
        if primary and primary != "Latn":
            continue
        selected.append(fam)
    log.info(
        "Selected top %d Latin-primary Sans/Serif/Display families",
        len(selected),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    allowlist: list[str] = []
    skipped: list[tuple[str, str]] = []
    file_count = 0
    for fam in selected:
        family = fam["family"]
        slug = family_to_slug(family)
        fdir = find_family_dir(args.google_fonts_repo, slug)
        if fdir is None:
            skipped.append((family, "directory not found"))
            continue

        family_stem = family_canonical_stem(family)
        variable_ttf = _variable_upright_ttf(fdir)
        produced: list[Path] = []

        if variable_ttf is not None:
            for weight, name in weight_pairs:
                out_path = _instance_variable(
                    variable_ttf, family_stem, weight, name, args.out_dir
                )
                if out_path is not None:
                    produced.append(out_path)
        else:
            for src, weight_name in _static_matches(fdir, weight_names):
                dest = args.out_dir / f"{family_stem}-{weight_name}.ttf"
                shutil.copy(src, dest)
                produced.append(dest)

        if not produced:
            skipped.append((family, "no upright weights matched"))
            continue

        file_count += len(produced)
        allowlist.append(family_stem)

    log.info(
        "Wrote %d .ttf files for %d families into %s",
        file_count, len(allowlist), args.out_dir,
    )
    if skipped:
        log.info("Skipped %d families:", len(skipped))
        for fam, reason in skipped[:15]:
            log.info("  - %s: %s", fam, reason)
        if len(skipped) > 15:
            log.info("  ... and %d more", len(skipped) - 15)

    literal = (
        "FONT_ALLOWLIST = [\n"
        + "\n".join(f'    "{f}",' for f in sorted(set(allowlist)))
        + "\n]\n"
    )
    if args.allowlist_out:
        args.allowlist_out.write_text(literal)
        log.info("Wrote allowlist to %s", args.allowlist_out)
    else:
        print(literal)

    return 0


if __name__ == "__main__":
    sys.exit(main())

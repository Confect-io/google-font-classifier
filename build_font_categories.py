#!/usr/bin/env python3
"""Generate the design-agent's `ocr/font_categories.json` for this vocabulary.

The design agent groups detected text regions by typographic category and
picks one font per category, so a family with no category can win a vote but
never be placed — `test_every_classifier_family_has_a_category` fails loudly
on any gap. That file was hand-curated for v4's ~33 stems; at 153 families
that stops being reasonable, and Google already publishes the answer.

Keys must be exactly what `font_db_mapping._label_to_family_stem()` returns
for each label, which for our bare family-stem labels is the stem itself.
Verified below rather than assumed.

Existing entries are merged, not replaced, so the file stays valid for the
currently deployed model until the swap happens.

Run:
    uv run --with requests python3 build_font_categories.py \\
        --labels font_labels_v6.json --out font_categories_v6.json
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

_METADATA_URL = "https://fonts.google.com/metadata/fonts"

# The five values design_agent/tests/test_font_db_mapping.py allows.
ALLOWED = {"serif", "sans-serif", "handwritten", "monospace", "display"}

# Google's `category` maps cleanly onto them. We deliberately keep Display as
# its own category rather than resolving it to sans/serif via the `stroke`
# field: the design agent wants a display headline and body copy treated as
# two different typefaces, which is what separate categories give it.
FROM_GOOGLE = {
    "Sans Serif": "sans-serif",
    "Serif": "serif",
    "Display": "display",
    "Handwriting": "handwritten",
    "Monospace": "monospace",
}


def stem(family: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", family)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, default=Path("font_labels_v6.json"))
    ap.add_argument("--merge", type=Path, default=None,
                    help="Existing font_categories.json to merge on top of")
    ap.add_argument("--out", type=Path, default=Path("font_categories_v6.json"))
    args = ap.parse_args()

    raw = json.loads(args.labels.read_text())
    labels = [raw[str(i)] for i in range(len(raw))]
    print(f"{len(labels)} labels from {args.labels}")

    with urllib.request.urlopen(_METADATA_URL) as r:
        meta = json.load(r)
    by_stem = {stem(f["family"]): f for f in meta["familyMetadataList"]}

    out: dict[str, str] = {}
    if args.merge and args.merge.is_file():
        out.update(json.loads(args.merge.read_text()))
        print(f"merged {len(out)} existing entries from {args.merge}")

    missing: list[str] = []
    for label in labels:
        fam = by_stem.get(label)
        if fam is None:
            missing.append(label)
            continue
        cat = FROM_GOOGLE.get(fam.get("category", ""))
        if cat is None:
            missing.append(label)
            continue
        out[label] = cat

    if missing:
        print(f"\nNO CATEGORY for {len(missing)}: {missing}")
        print("Add these by hand before shipping — the design-agent test fails on gaps.")

    bad = set(out.values()) - ALLOWED
    assert not bad, f"produced categories outside the allowed set: {bad}"

    args.out.write_text(json.dumps(dict(sorted(out.items())), indent=2) + "\n")
    counts: dict[str, int] = {}
    for v in out.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"\nwrote {len(out)} entries to {args.out}")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<12} {v}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

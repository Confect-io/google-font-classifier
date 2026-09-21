#!/usr/bin/env python3
"""Fill the font vocabulary out from a human-picked seed set.

The hand-picked families are LOCKED — they are the ones we know we need.
Everything added on top is chosen on two axes only:

  rank        walk candidates in popularity order, most-used first
  uniqueness  skip any family within --min_distance of something already
              in the set, so the fill never adds a twin

Uniqueness is a FILTER, not the objective. Optimising for distinctness
directly (greedy farthest-point) puts Libre Barcode 39, Eater and Monoton at
the top of the list: maximally unlike everything else, and never used to set
a word of ad copy. Popularity picks what is worth knowing; the distance floor
stops it spending classes on faces the model could not tell apart anyway.

Run:
    uv run --with numpy python3 fill_vocabulary.py \\
        --seed_dir <dir of picks/*.json> \\
        --target 250 --min_distance 0.006 \\
        --out ./vocabulary_proposal.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

import numpy as np

CAT_ORDER = ("sans-serif", "serif", "display", "handwritten", "monospace")

# Families that are not typefaces for setting words. Google files them under
# Display, and pure distinctness scoring loves them precisely because they look
# like nothing else -- Libre Barcode 39 came top of an early run. No ad sets
# copy in a barcode, so they never belong in the vocabulary.
NON_TEXT = ("barcode", "symbol", "emoji", "icon", "dingbat", "music",
            "chess", "braille", "semaphore")


def is_non_text(family: str) -> bool:
    low = family.lower()
    return any(k in low for k in NON_TEXT)


def load_seed(seed_dir: Path) -> set[str]:
    stems = set()
    for f in glob.glob(str(seed_dir / "*.json")):
        stems.add(os.path.basename(f)[:-5])
    return stems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed_dir", type=Path, required=True,
                    help="Directory of one JSON per hand-picked family (id = stem)")
    ap.add_argument("--candidates", type=Path, default=Path("./picker_data.json"))
    ap.add_argument("--embeddings", type=Path, default=Path("./font_embeddings.npz"))
    ap.add_argument("--target", type=int, default=250,
                    help="Total families in the finished vocabulary")
    ap.add_argument("--rank_gate", type=int, default=10**9,
                    help="Hard popularity cutoff; the target is normally what stops it")
    ap.add_argument("--min_distance", type=float, default=0.006,
                    help="Skip a family this close to one already in the set. "
                         "0.006 is roughly the Roboto/Open Sans gap — true twins.")
    ap.add_argument("--out", type=Path, default=Path("./vocabulary_proposal.json"))
    args = ap.parse_args()

    data = json.loads(args.candidates.read_text())
    members: dict[str, dict] = {}
    for c in data["clusters"]:
        for m in c["members"]:
            members[m["s"]] = {**m, "category": c["category"], "cluster": c["id"],
                               "cluster_size": c["size"]}

    z = np.load(args.embeddings, allow_pickle=True)
    emb, names = z["embeddings"], [str(n) for n in z["families"]]
    fam_to_row = {n: i for i, n in enumerate(names)}
    # The payload is keyed by stem, the embeddings by display name.
    stem_to_row = {m["s"]: fam_to_row[m["f"]] for m in members.values()
                   if m["f"] in fam_to_row}

    seed = load_seed(args.seed_dir)
    seed = {s for s in seed if s in members}
    print(f"Seed (locked):     {len(seed)} families")

    selected = list(seed)
    sel_rows = [stem_to_row[s] for s in selected if s in stem_to_row]

    pool = [s for s, m in members.items()
            if s not in seed and m["p"] <= args.rank_gate and s in stem_to_row
            and not is_non_text(m["f"])]
    dropped = [m["f"] for s, m in members.items()
               if s not in seed and m["p"] <= args.rank_gate and is_non_text(m["f"])]
    print(f"Pool (rank <= {args.rank_gate}): {len(pool)} candidates")
    if dropped:
        print(f"  excluded as non-text: {', '.join(sorted(dropped))}")
    print(f"Target:            {args.target}\n")

    # Walk the pool most-popular-first; keep a running nearest-distance to the
    # selected set so a candidate is measured against earlier additions too,
    # not just the seed.
    pool.sort(key=lambda s: members[s]["p"])
    pool_rows = np.array([stem_to_row[s] for s in pool])
    pool_emb = emb[pool_rows]

    sims = pool_emb @ emb[sel_rows].T
    nearest = 1.0 - sims.max(axis=1)
    # Track WHICH selected family is the nearest, so a skip can name the font
    # that actually blocked it rather than some unrelated pool neighbour.
    blocker = [selected[j] for j in sims.argmax(axis=1)]

    added: list[tuple[str, float]] = []
    skipped: list[tuple[str, float, str]] = []

    for i, stem in enumerate(pool):
        if len(selected) >= args.target:
            break
        if nearest[i] < args.min_distance:
            skipped.append((stem, float(nearest[i]), blocker[i]))
            continue
        added.append((stem, float(nearest[i])))
        selected.append(stem)
        d_new = 1.0 - (pool_emb @ emb[stem_to_row[stem]])
        closer = d_new < nearest
        nearest = np.where(closer, d_new, nearest)
        for j in np.flatnonzero(closer):
            blocker[j] = stem

    print(f"Added:             {len(added)}")
    print(f"Skipped as twins:  {len(skipped)}")
    print(f"Final vocabulary:  {len(selected)}\n")

    cats = collections.Counter(members[s]["category"] for s in selected)
    seed_cats = collections.Counter(members[s]["category"] for s in seed)
    print(f"{'category':<13} {'seed':>5} {'final':>6}")
    for c in CAT_ORDER:
        print(f"  {c:<11} {seed_cats.get(c,0):>5} {cats.get(c,0):>6}")

    ranks = sorted(members[s]["p"] for s in selected)
    print(f"\npopularity: median rank {ranks[len(ranks)//2]}, worst {ranks[-1]}")
    singles = sum(1 for s in selected if members[s]["cluster_size"] == 1)
    print(f"visually distinct (singleton clusters): {singles}")

    print(f"\nfirst 20 added (most popular first):")
    for stem, d in added[:20]:
        m = members[stem]
        print(f"  rank {m['p']:>4}  dist {d:.3f}  {m['f']:<26} {m['category']}")
    print(f"\nlast 5 added (the tail this target reaches):")
    for stem, d in added[-5:]:
        m = members[stem]
        print(f"  rank {m['p']:>4}  dist {d:.3f}  {m['f']:<26} {m['category']}")
    # The consequential calls: popular families the fill left out. These are
    # where a human should overrule the rule, so name the blocker explicitly.
    print(f"\nTOP-100 FAMILIES NOT IN THE VOCABULARY (review these):")
    sel_set = set(selected)
    left_out = sorted(
        (m for s, m in members.items() if s not in sel_set and m["p"] <= 100),
        key=lambda m: m["p"],
    )
    skip_by_stem = {s: (d, b) for s, d, b in skipped}
    for m in left_out:
        d, b = skip_by_stem.get(m["s"], (None, None))
        why = (f"twin of {members[b]['f']} at {d:.4f}" if b
               else "below the target cutoff")
        print(f"  rank {m['p']:>4}  {m['f']:<24} {why}")
    if not left_out:
        print("  (none — every top-100 family made it in)")

    payload = {
        "target": args.target,
        "rank_gate": args.rank_gate,
        "seed_count": len(seed),
        "added_count": len(added),
        "families": sorted(members[s]["f"] for s in selected),
        "seed": sorted(members[s]["f"] for s in seed),
        "added": [
            {"stem": s, "family": members[s]["f"], "category": members[s]["category"],
             "cluster": members[s]["cluster"], "rank": members[s]["p"],
             "distance": round(d, 4)}
            for s, d in added
        ],
    }
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

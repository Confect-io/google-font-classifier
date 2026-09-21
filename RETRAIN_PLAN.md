# v6 Retrain Plan

Working plan for the next classifier retrain. Review this before any code
changes. Background on how the pipeline runs is in `CONFECT.md`; this doc
only covers what changes and in what order.

## Why

v5 (147 families / 266 classes, Regular+Bold) finished 100 epochs on an
RTX 4090 in ~11h and peaked at **96.65%** validation accuracy — against the
paper's 99.0% on 32 hand-picked families. The gap is the font list, not the
model: `curate_fonts.py` selects on **popularity alone**, which

- packs the set with mutually indistinguishable neutral grotesks (the top 20
  Latin families are 18 sans-serifs — Inter, Archivo, Figtree, Manrope,
  Public Sans, Schibsted Grotesk, Onest, Hanken Grotesk, Albert Sans, ...), and
- filters out Handwriting entirely (`_CATEGORY_ALLOWED` is Sans/Serif/Display),
  so there are zero script/cursive fonts in the vocabulary.

Available to draw from: **1,248 Latin-primary families** — 390 sans, 383
display, 216 handwriting, 211 serif, 48 mono.

## Decisions (locked)

| Decision | Choice |
|---|---|
| Selection basis | **Opt-in against a cap, curated by humans.** A reviewer clicks the families to include; the picker refuses additions past the max. No anchoring to Confect's catalogue — the catalogue is being backfilled, so the classifier's vocabulary becomes the binding constraint, not catalogue coverage. |
| Class scheme | **Family-only.** One class per family; weights 300–700 rendered *within* the class as augmentation. |
| Candidate pool | **964 families** — the top ~250 by popularity in each of the five categories, Latin-primary. |
| Vocabulary | **153 families — LOCKED.** 83 hand-picked + 70 filled by rank ≤ 200 with true siblings pruned. Lives in the picker's store and in `vocabulary_proposal.json`. |
| LoRA rank | **r=16** (`--mode lora16`), not r=8. |

## What this model is for

Stated plainly, because it settles several arguments below: **the model does
not need to name the exact font.** An ad's type needs to resolve to something
that looks very close and reads right. `fonts.py` walks the top 5 and takes
the first family Confect's catalogue carries, so Inter-for-Public-Sans is a
correct outcome, not an error.

Two consequences:

- **Coverage of fonts people actually use beats precision.** Rank ordering is
  the right selection axis; visual distinctness is only a tie-breaker that
  stops the vocabulary spending classes on identical siblings.
- **The vocabulary is not to be maxed out.** Every family in it is a family
  the model can *output*, so an obscure one is an obscure answer. A detected
  font should always land on something a designer recognises. That is why the
  gate is rank 200, not 300: the 200–300 band is Viga, Quantico, Alata, Kumbh
  Sans, Hammersmith One, Advent Pro — nobody wants those as the result.
- **Script and display are a minor concern.** Ads are overwhelmingly set in
  sans and serif, so a vocabulary that is half sans-serif is correct, not
  skewed. The earlier worry about category balance does not apply.

It also means the fine-discrimination work — 448px input, tile-and-vote,
`dinov2-large` — is **optional, not required**. Those raise precision on
near-twins, which is explicitly not what this model is judged on. Keep them
in reserve; do not let them gate the retrain.

**Clusters do not prune.** An earlier draft had the tooling pick a diverse
subset and drop the lookalikes. That is wrong on both counts: a font the model
has never seen is a font the design agent can never name, and optimising for
distinctness surfaces exactly the obscure faces we do not want returned.
Clustering earns its place as *review context* — it shows a reviewer where the
model will trade guesses between interchangeable faces — and as a **twin
filter** at 0.002, which removes only genuine siblings. It is not a ranking.

## What actually constrains the vocabulary

Not class count. The classifier head is 768×N — 740K parameters at 964
classes, on an 87M-parameter model; this backbone class routinely handles
ImageNet-21k's 21,841 classes. Nothing about 250 vs 900 fonts strains it.

What costs accuracy is **cluster density**, and the shortlist puts numbers on
it: 73 clusters with spread < 0.02 hold **413 of the 964 candidates** — that
is where every confusion will concentrate — while **170 families are
singletons** that nothing else resembles. Adding the 36th grotesk to the
Roboto/Open Sans/Inter blob costs a little, and costs it only inside that
blob. A singleton is free *in accuracy terms* — but most singletons are
obscure, and an obscure class is an obscure answer, so cheapness is not a
reason to add one. Popularity decides; distance only breaks ties.

The one genuine capacity question is **LoRA rank**. r=8 on query/value is
~600K trainable parameters. The paper found r=4/8/16 statistically
indistinguishable at 394 classes, so rank was not binding there; at 153
family-only classes it almost certainly is not either — r=16 measured 98.9% against
r=8's 99.0% — noise — so **run `--mode lora16`**: it doubles adaptation
capacity for no measured accuracy cost.

The binding constraint is therefore neither capacity nor cost — it is
**recognisability**. A family in the vocabulary is a family the model can
return, so the gate sits where the names stop being ones a designer knows:
rank 200.

Worth keeping in proportion: widening 147 → 153 families barely moves
*coverage*; what changes is the composition (script and display now present,
identical siblings gone). Whether the model reads any of them correctly **on
real ads** is a domain-gap problem, and the tier-1 augmentation fixes below
are by far the weaker link — the existing fixtures already miss Poppins and
PT Serif on real creative with only 147 families.

This has a consequence worth stating plainly: **raw top-1 accuracy will drop**,
and that is acceptable. v5 fell to 96.65% largely because of near-identical
sans-serifs, and widening the vocabulary pushes it further. The error is
nearly free — confusing Public Sans with Inter costs the design agent almost
nothing, because the two are visually interchangeable and `fonts.py` reads the
top 5 and takes the first family the catalogue carries. Judge this model on
the **visual-severity-weighted metric** (`compute_swer.py`, already in the
repo), not on top-1.

Rationale for family-only: downstream `_label_to_family_stem()` strips the
weight off every label and `DetectedFont.entry` is just `{id, name,
source_key}` — **the predicted weight is discarded**. v5 spent half its class
budget, dataset, and confusion on a distinction nothing consumes. Folding
weight into the class buys ~2x family coverage for free *and* removes
within-family weight confusion, which the paper names as the dominant error
mode. Cost: this model can never report weight. Stroke-width heuristics on
the crop can recover that separately if it's ever needed.

## Steps

### 1. Human curation

Machines narrow and organise; humans pick. Three parts.

#### 1a. Shortlist + cluster (tooling)

Nobody can review 1,248 families in a flat alphabetical list, and that is
exactly how you end up with fourteen near-identical grotesks. So:

1. Fetch `fonts.google.com/metadata/fonts`; filter to Latin-primary, all five
   categories (Handwriting and Monospace now included).
2. Drop the long tail by popularity (keep ~top 250 per category), so the
   review set is fonts people actually use.
3. Embed each candidate: render 3–4 fixed strings (uppercase pangram,
   lowercase sentence, digits/currency) in the family's Regular weight,
   black-on-white, no noise; push through **DINOv2-base itself** — the exact
   backbone being trained, so similarity is measured in the space the model
   learns in — and average the L2-normalised CLS vectors.
4. Cluster the embeddings. The clusters are **not** picks — they are the
   review layout. A cluster of 14 lookalike grotesks becomes one screen that
   says "these are near-identical to the model; keep 2."

#### 1b. The picker (human) — BUILT

**https://claude.ai/artifact/2JcfGm6PKkrovarsYJZTTH**

A specimen sheet, not a spreadsheet. Google Fonts serves every one of these
as a webfont, so the browser renders real specimens with no asset pipeline.

- One card per family: name, popularity rank, its nearest lookalike and the
  distance to it, and a specimen line set in the actual font. A face whose
  webfont fails to load is flagged rather than shown in a fallback — judging
  a specimen rendered in the wrong font would be a silently wrong review.
- Specimen text switches between a type-specimen string, the alphabet, ad
  copy and prices — the last two being what this model actually reads.
- Grouped by **similarity cluster**, so interchangeable faces sit side by
  side and the redundancy is legible. The most-used face leads each cluster,
  and clusters run most-popular-first by default (toggle to most-crowded-first
  to go redundancy-hunting).
- **Opt-in.** Nothing is included until clicked; adding past the cap is
  refused with a prompt to remove one or raise the max. Per-cluster
  "Mark reviewed" tracks coverage across all 300 clusters.
- **Tracker** on the header: included against the cap, slots remaining, a
  meter that turns amber at the limit, per-category counts, and clusters
  reviewed.
- Live **cost readout** scaled from the measured v5 run (266 classes × 615
  images = ~11 h at 100 epochs on a 4090, ~40 GB): image count, dataset size,
  training hours and GPU cost at 40 or 100 epochs. This is what makes "how
  many fonts" a decision with a number attached rather than a guess.
- Shared and live: the store is organization-internal, so any signed-in
  member of the Claude team who opens the link can review, and everyone's
  drops sync to everyone else.

#### 1c. What has to come back

Nothing, in the usual sense — **the picks are read straight out of the
page's shared store**, so there is no file to hand over and no risk of an
out-of-date copy. The page also exports the same thing on demand:

```json
{ "families": ["Open Sans", "Playfair Display", "Dancing Script", "..."] }
```

Exact Google Fonts family names as they appear in Google's metadata
(`"Open Sans"`, `"PT Sans Narrow"` — spaces, original casing). That is the
contract; everything downstream keys off it. A newline-separated list of the
same names works too if the picker is ever bypassed.

On ingest, the tooling validates every name against Google's metadata and
the local `google/fonts` clone, and reports anything that cannot be resolved
to a real family directory rather than silently dropping it.

#### 1e. The fill — `fill_vocabulary.py` (DONE)

The 83 hand-picked families were the must-haves; the rest was filled
mechanically:

```
uv run --with numpy python3 fill_vocabulary.py \
    --seed_dir <picks> --target 964 --rank_gate 200 --min_distance 0.002
```

Walk candidates most-popular-first, skip any family within `min_distance` of
one already in the set. **Uniqueness is a filter, not the objective** — an
early version optimised distinctness directly and put Libre Barcode 39,
Eater and Monoton at the top, faces that are maximally unlike everything
else and never set a word of ad copy. `NON_TEXT` now excludes barcode and
symbol families outright.

`min_distance` is **0.002**, not the 0.006 that looks like the natural "twin"
cutoff. At 0.006 the rule dropped Ubuntu as a twin of Outfit and Plus Jakarta
Sans as a twin of Comfortaa — wrong to any designer. The embedding reads a
whole rendered word, so overall texture and weight dominate letterform
detail; it is trustworthy for true siblings and noisy above ~0.003. At 0.002
the only top-100 exclusions are Nunito Sans (vs Nunito), Lexend (vs Lexend
Deca, distance 0.0000) and Roboto Flex (vs Roboto) — all genuinely the same
design.

Result: **153 families** — 83 sans, 29 serif, 14 display, 18 script, 9 mono.
Median popularity rank 118.

The gate landed at 200 after looking at what each one admits: ≤150 gives 124
families, ≤200 gives 153, ≤250 gives 176, ≤300 gives 213. The 200–300 band is
where the names stop being recognisable, so that is the cut.

#### 1d. `curate_fonts.py` → consumer, not selector

It stops choosing fonts. It reads the picked list and instances each family
to weights **300/400/500/600/700** (keeping whatever the family supports,
minimum one) into `fonts/<FamilyStem>/<FamilyStem>-<Weight>.ttf` — a
directory per family, because the directory is now the class.

It also emits `font_categories.json` for the design agent straight from the
metadata (`category`/`stroke` → `sans-serif`/`serif`/`display`/`handwritten`/
`monospace`). Downstream already accepts all five values, so this replaces a
hand-curated, test-enforced file with a generated one.

### 2. `dataset_generator.py` → family classes

- Class = subdirectory name under `--font_dir`. **Delete the hardcoded
  `FONT_ALLOWLIST`** and the manual copy-paste step with it.
- Per image, pick a random weight from the family's TTFs. Keep ~575 train / 40
  test per class, so the per-class data budget matches v5.

## Separating the lookalikes

The stated priority is that the model not confuse near-identical faces. The
bottleneck there is **pixels per glyph, not model capacity** — and that is
good news, because it is fixable.

Trace the current geometry. A training image is a multi-line block rendered
and resized to 256 tall, then padded to square and resized to 224: content
occupies roughly 224×112, each line ~37 px, cap height **~18 px**. At
inference it is worse — OCR hands `fonts.py` a single text line, often 10:1,
which pads to square and lands at cap height **~12 px**. Telling Inter from
Public Sans at 12 px of cap height is not a capacity problem. The
information is not in the pixels.

Levers, most effective first:

1. **Raise the input resolution, 224 → 448.** DINOv2 interpolates its
   position embeddings, so this needs no architecture change — 4× the glyph
   detail for ~4× the compute, which is explicitly not a constraint here.
   This is the single biggest lever and it helps exactly the confusions that
   matter.
2. **Tile-and-vote at inference.** A 10:1 crop cut into ~4 square tiles,
   each classified at full resolution and the logits averaged, gives 4× the
   glyph detail *and* four votes. This is a `fonts.py` change with **no
   retraining at all** — worth trying against the real-ad fixtures before
   anything else on this list.
3. **Train on single-line crops** so training geometry matches what OCR
   actually produces (tier-1 augmentation #1 below).
4. **`dinov2-large`** (300M) in place of `dinov2-base` (86M).
5. **Wider LoRA.** r=16 as decided, and target more than `["query", "value"]`
   — adding `key` and the MLP `dense` layers materially increases what the
   adapter can reshape.
6. **Oversample the tight clusters.** `font_candidates.json` already carries
   the cluster structure; weighting those 73 dense clusters higher in the
   sampler spends gradient exactly where the model is confusable.

Worth being straight about the tension: more fonts and better lookalike
accuracy pull against each other, but *only inside the dense clusters*.
Singletons and distinct faces are free. So be generous with the 170
singletons and selective inside the crowded blobs — which is precisely what
the picker's spread and twin-distance readouts are for.

### 3. Augmentation — tier 1 only

Fix the three that are genuine train/serve skew, leave the rest for later:

1. **Aspect ratio.** Training renders multi-line wrapped blocks at ~2:1–4:1;
   production feeds single-line OCR crops, often 10:1+, which after
   pad-to-square become a thin stripe in a mostly-black square. Bias generation
   heavily toward single-line.
2. **Resolution.** Everything is rendered at `font_size=1024` and downsampled —
   pristine. Real crops are 20–40px tall and *upsampled*. Add a downsample/
   upsample round-trip.
3. **JPEG artifacts.** Every production input has been through JPEG at least
   once; no training image has.

Deferred: photo/gradient backgrounds, blur, rotation/shear, letter-spacing,
ad-copy corpus (currently 18th-century prose, no ALL CAPS headlines), noise
sampled across a range instead of fixed σ=0.1·255.

### 4. Generate + upload

`dataset_generator.py` → `dataset_cleaner.py` → tar → `hf upload` to
`confect/google-font-dataset`. Bump to a v6 repo or branch so v5 stays
reproducible.

**Switch the image format to JPEG first.** This is the change that makes the
whole scale question go away, and it is two lines.

`dataset_generator.py:231` adds Gaussian noise at σ=25.5 to every pixel, and
`:273` saves PNG at `compress_level=1`. Noise is nearly incompressible and
level 1 barely tries, so every image costs **250 KB**. Measured on the real
v5 output:

| Format | Per image | 153 families × 615 |
|---|---|---|
| PNG `compress_level=1` (today) | 250 KB | **24 GB** |
| JPEG q=95 | 65 KB | 6.1 GB |
| JPEG q=90 | 48 KB | 4.5 GB |
| JPEG q=80 | 32 KB | **3.0 GB** |
| JPEG q=70 | 24 KB | 2.3 GB |

Save JPEG at a **randomised quality (roughly 60–95)** and the dataset drops
~8×, *and* it delivers tier-1 augmentation #3 for free — every image the
model sees in production has been through JPEG at least once, and not one
training image currently has. The compression artifacts are not a cost here,
they are the point.

Consequences:

- **Generation on a laptop is a non-issue.** 153 families is ~3 GB and
  ~4 minutes at v5's measured rate (266 classes in ~7 min, 8 workers).
- **No HuggingFace plan change.** 3 GB is a fraction of the 43 GB already
  sitting in the private `confect/google-font-dataset`.
- **It also dodges a hard limit.** HF caps LFS files at 50 GB, which the
  old PNG pipeline would have hit at a larger vocabulary. If a tar ever does approach that,
  shard it — `cloud_train.sh` already globs `data/train*.tar`, so
  `train00.tar` / `train01.tar` extract with no code change.

**The real scaling wall is on the training box, not the laptop.**
`train_model.py` calls `dataset.map(transform)`, which materialises every
image as a float32 tensor into an Arrow cache — 3×224×224×4 = 602 KB each:

| Classes | Arrow cache |
|---|---|
| 266 (v5) | ~92 GB |
| 153 | ~53 GB |

`cloud_train.sh`'s TODO calls this a "1-2 hour cache rebuild". **Measured from
the v5 log, it is ~25 minutes** (152,950 images at ~110 examples/sec:
`15:10:13 Applying data transformations` -> `15:35:01 Data preprocessing
complete`). At 252,450 images that scales to **~38 minutes**, paid on every
instance launch. Worth removing, but it is a convenience fix, not a cost
problem — do not let it gate a run.

Fix: swap `.map()` for **`set_transform()`**, which applies the transform
lazily per batch — zero extra disk, no rebuild, and the CPU cost overlaps
with GPU compute under the existing `dataloader_num_workers=4`. This edits
`train_model.py`, so it depends on step 5.

### 5. Point cloud training at our fork

`cloud_train.sh:263` clones **`Create-Inc/font-model`**, not this repo — local
changes to `train_model.py` / `handler.py` never reach the GPU. Repoint it at
`Confect-io/google-font-classifier` before any train-time change matters.

### 6. Train

Dry run first (`--dry_run`, ~5 min, ~$0.05), then the real run. v5's curve:
93.4% @ ep10 → 95.2% @ ep30 → 95.9% @ ep40 → 96.6% @ ep100. Epochs 40→100
bought 0.8 points for 6 extra hours, so **run 40 epochs first** and only
extend if the number justifies it — at 964 classes that is ~16 h and ~$9,
against ~40 h and ~$22 for the full 100.

### 7. Export

`export_onnx.py` — update label derivation from sorted `.ttf` stems to sorted
**directory** names. Produces the ONNX + `font_labels.json`.

### 8. Catalogue backfill (parallel workstream)

Ingest every missing Google Font into Confect's `fonts` table. Today
`font_db_mapping.json` carries **14 families**, so most of what the classifier
knows is unusable downstream. `build_font_db_mapping.py` matches by normalised
name, so once the table carries the same family names as Google Fonts, the
mapping regenerates with no code change:

```
\copy (SELECT id, name, ... FROM fonts WHERE account_id IS NULL)
  TO '~/Downloads/font_google_dump.csv' CSV HEADER;
uv run python -m design_agent.scripts.build_font_db_mapping ~/Downloads/font_google_dump.csv
```

### 9. Deploy

Upload ONNX to `cdn.confect.io/static/font-classifier-v6.onnx`; update
`_DEFAULT_MODEL_URL`, the Dockerfile `ADD --checksum=sha256:`,
`font_labels.json`, `font_categories.json` and `font_db_mapping.json` in
`design_agent`; validate with `pytest tests/test_fonts.py -v` against the
real-ad fixtures (out-of-distribution, so the meaningful signal — in-dist
validation accuracy is not predictive).

**Note:** production is still on **v4** (`font-classifier-v4.onnx`, 394 labels)
on both `main` and `development`. v5 was exported but never landed in the
design agent.

## Open items for review

- [ ] **Images per class.** At 153 classes the dataset is small enough
      (~94k images, ~3 GB as JPEG) that 575/class is comfortable — and with
      fewer classes than v5 there is an argument for raising it instead, since
      each family now has to cover five weights. Worth a look before
      generating.
- [ ] Do tier-1 augmentation in the same pass as the vocabulary change, or
      change one variable at a time so each effect is attributable?
- [ ] Italics: currently excluded entirely. Worth adding as in-class
      augmentation now that weight is already an in-class axis?
- [ ] Switch the reported metric to visual-severity-weighted error
      (`compute_swer.py`) before the run, so the result is judged on the right
      axis from the start rather than compared against v5's 96.65% top-1.
- [ ] Do tier-1 augmentation in the same pass as the font-list change, or
      change one variable at a time so the effect of each is attributable?
- [ ] Italics: currently excluded entirely. Worth adding as in-class
      augmentation now that weight is already an in-class axis?

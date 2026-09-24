# Font classifier v2: family and weight prediction

Status: proposed implementation plan

Tracking: Confect's design agent currently uses the v6 family classifier plus
a conservative morphology-based weight hint. That is v1. This document defines
v2: retrain the classifier with an independent weight output that also works on
font families outside the classifier vocabulary.

## Goal

For every OCR line, predict:

1. the closest supported family from the existing 153-family vocabulary; and
2. a family-independent upright weight group with calibrated confidence.

Do not create a class for every family and weight combination. A custom font
must be able to receive a weight prediction even though it has no family class.

The product mainly needs neighboring CSS weights to be interchangeable:

| Group | CSS weights | Product meaning |
|---|---|---|
| ultra-light | 100, 200 | thin custom variants |
| light/regular | 300, 400 | visually light or normal |
| medium/semibold | 500, 600 | visibly stronger than regular |
| bold/heavy | 700, 800, 900 | clearly bold |

The required promotion evaluation covers 300-900. Train and report 100/200 as
a separate group so custom thin variants are not silently reported as regular,
but do not let sparse ultra-light coverage decide the initial promotion. Do not
include italic classification in this work.

## Relationship to v1

The v1 design-agent integration remains the production fallback while v2 is
trained and evaluated. It measures crop morphology after the family is known
and emits two advisory candidates only when its calibration is strong enough.
Its locked realistic result was 93.5% group precision at 21.4% coverage.

V2 must replace v1, not run as a second competing hint. Once v2 passes its
promotion gates and is integrated, delete the morphology estimator and its
calibration artifact in the same change. If v2 fails the gates, keep v1 and do
not deploy the new weight output.

## Model design

Use the existing DINOv2 backbone and two independent heads:

```text
preprocessed OCR crop
        |
  DINOv2 features
     /       \
153-family   ordinal weight
   head          head
```

The family head stays exactly 153-way. The weight head is ordinal rather than
an ordinary softmax because the groups have an order and adjacent errors are
less severe than distant errors.

Use three cumulative logits for the four groups:

- `P(weight >= 300)`
- `P(weight >= 500)`
- `P(weight >= 700)`

The exported metadata must state the thresholds so runtime code does not
hardcode the training setup.

The ONNX model exposes:

```text
family_logits: [batch, 153]
weight_logits: [batch, 3]
```

Do not add an embedding output in the first implementation. The existing
custom-family matcher may continue comparing family logits. A dedicated
embedding can be evaluated later without blocking the weight head.

## Dataset changes

`dataset_generator.py` currently chooses a random font file and then stores
only the family directory as the label. Preserve the chosen variant as sample
metadata:

- family label, or a masked value for weight-only families;
- numeric weight from `OS/2.usWeightClass`;
- derived weight group;
- source family identifier;
- rendered text and augmentation seed when practical for reproducibility.

Validate the numeric label against font metadata. Filenames may be used for a
readable variant name but must not be the source of truth.

Balance weight groups within each family before balancing individual weights.
Otherwise variable families with many files can dominate the weight loss.
Continue varying text, size, antialiasing, compression, crop slack, color, and
background. Include flat, gradient, textured, and photographic backgrounds.

Build three sources:

1. **Known families:** the 153 classifier families, with family and weight
   supervision.
2. **Weight-only families:** fonts outside the 153-family vocabulary, with
   weight supervision and family loss masked. These teach the head not to rely
   on a known family identity.
3. **Realistic evaluation crops:** full designs passed through RapidOCR and
   cropped exactly as production does. Keep these outside training.

Customer-uploaded font files must not be copied into a training corpus without
an explicit data and licensing decision. Public fonts outside the 153-family
vocabulary are sufficient for the initial unseen-family benchmark.

## Splits that demonstrate custom-font generalization

Random image splits are insufficient: the model could learn each family's
typical stroke widths and appear to generalize.

Use these disjoint splits:

- normal train/validation/test images for the 153-way family task;
- weight calibration and test crops disjoint by text and augmentation seed;
- external weight-only families split **by family**, so final-test families
  never contribute weight supervision during training;
- a locked set of custom-like fonts that are absent from the 153 family labels
  and from the weight-training family set.

Report known-family and unseen-family weight results separately. The
unseen-family result is the primary custom-font metric.

## Training sequence

### Phase 1: frozen probe

Load the deployed v6 checkpoint, freeze the backbone and existing family head,
and train only the ordinal weight head. This is the cheapest experiment and
guarantees that family predictions cannot regress.

### Phase 2: joint LoRA training

Only run this phase if the frozen probe misses the weight gate. Train with:

```text
loss = family_cross_entropy + lambda_weight * ordinal_weight_loss
```

Mask `family_cross_entropy` for weight-only families. Apply the weight loss to
every sample with reliable weight metadata. Tune `lambda_weight` on validation
data, and retain the existing family checkpoint when two runs are equivalent.

Save both heads in the LoRA checkpoint. Select checkpoints using a composite
validation report that cannot hide a family regression behind improved weight
accuracy.

CORAL or CORN are suitable ordinal formulations. Start with CORAL because the
head is small and simple; change it only if validation demonstrates a material
benefit.

## Evaluation and promotion gates

For family prediction, report the existing top-1, top-5, and
visual-severity-weighted metrics on the current locked family evaluation set.
V2 must not regress top-1 by more than 0.5 percentage points or materially
worsen the severity-weighted result.

For weight prediction, report:

- exact group accuracy and macro accuracy across families;
- confusion matrices;
- calibrated precision/coverage curves;
- results by known versus unseen family;
- results by text height and background type;
- abstention rate and reasons;
- accuracy after filtering to the selected font's supported weights.

Choose confidence thresholds on validation data and apply them unchanged to
the locked test sets. Promotion requires all of the following:

- at least 95% group precision at 50% or greater coverage for 300-900 crops;
- the same gate passes on the unseen-family/custom-like test set, not only on
  the 153 known families;
- no unacceptable family-classification regression;
- full-canvas RapidOCR evaluation confirms that OCR crop generation does not
  invalidate the tight-crop result.

Single-weight families may be reported separately but do not count toward the
weight gate. They are a metadata resolution, not evidence that the detector
works.

## Confidence calibration

Fit calibration on a dedicated split after model training. Calibrate the
probability of the selected group being correct, not the probability of an
exact numeric CSS weight.

Runtime states are:

- **authoritative:** confidence exceeds the locked promotion threshold;
- **advisory:** measurable but below the authoritative threshold;
- **abstained:** crop quality or confidence is insufficient.

Do not reuse thresholds chosen for known families on custom fonts unless the
unseen-family calibration demonstrates that they remain valid. Prefer one
conservative global threshold if it passes both populations; otherwise export
separate `known_family` and `unseen_family` thresholds.

## Custom-font runtime behavior

The custom-family matcher and the weight head solve different problems:

1. Run the model once per OCR line and retain both outputs.
2. Resolve the 153-way family palette as today.
3. Allow the custom matcher to replace a Google family with an uploaded font.
4. Resolve the already-computed weight probabilities against the selected
   font's actual upright variants.

Map a predicted group to concrete supported weights:

- one supported weight in the group: return it;
- multiple supported weights in the group: rank them as alternatives and let
  the design agent compare them visually;
- no supported weight in the group: return nearest supported alternatives as
  advisory, never invent an unsupported CSS weight;
- one upright variant in the entire custom family: return that variant as
  fixed, without model confidence;
- unreadable, tiny, or mixed-style line: abstain.

Predictions remain attached to OCR attachment and box identity. Never pool by
text string, category, text layer, or selected family because separate lines
often use different weights and duplicate strings are common.

The first implementation does not need to download every custom-font variant
for image comparison. Variant metadata is enough to filter the ordinal result.
Same-text rendering of all variants is a possible later refinement when exact
700-versus-800 selection becomes valuable.

## Design-agent integration

Deploy the model under a new versioned artifact name; do not overwrite v6.
Update the runtime wrapper to read both ONNX outputs and validate their shapes
against versioned label metadata.

Prompt behavior:

- authoritative group results constrain the selected weight to supported
  variants in that group;
- advisory results provide only supported candidates and require visual
  comparison with the matching OCR line;
- abstentions fall back to visual judgment;
- italic remains a visual decision.

Log only aggregate authoritative/advisory/abstained counts. Do not log crop
pixels or OCR text. No HTTP API or stored Design schema change is required.

## File-level implementation checklist

Classifier repository:

- `dataset_generator.py`: emit verified family and weight metadata.
- `train_model.py`: introduce the shared backbone, family head, ordinal head,
  masked losses, metrics, and frozen-probe mode.
- `export_onnx.py`: merge adapters and export both named outputs plus versioned
  weight metadata.
- `handler.py`: return family and calibrated weight results for hosted tests.
- add a deterministic evaluator for known-family, unseen-family, and realistic
  OCR sets.
- keep the existing family label order unchanged.

Confect repository:

- update `ocr/fonts.py` to batch both outputs per OCR line;
- resolve the weight group only after Google/custom family selection;
- filter candidates through catalogue or uploaded-font supported variants;
- update image and video prompt formatting;
- replace and delete the v1 morphology/calibration path only after promotion;
- keep OCR identity and duplicate-text coverage in tests.

## Required tests

- weight labels come from font metadata and reject mismatches;
- group boundaries and cumulative targets are deterministic;
- family loss is masked for external families while weight loss remains active;
- weight loss is masked when metadata is absent;
- ONNX and PyTorch outputs agree within tolerance;
- exported family labels, weight thresholds, and output shapes stay synchronized;
- known-family and unseen-family splits share no family IDs;
- supported-weight filtering never returns an unavailable variant;
- single-variant custom fonts return fixed rather than model-derived confidence;
- duplicate OCR text retains independent line predictions;
- authoritative, advisory, and abstained prompt formatting behaves identically
  in image and video modes;
- the v1 estimator and calibration artifact are absent once v2 is promoted.

## Deliverables

1. Reproducible dataset manifest and split seeds.
2. Frozen-probe results.
3. Joint-training results only if the probe is insufficient.
4. Locked evaluation report with known and unseen families separated.
5. Versioned ONNX model and compact metadata.
6. Design-agent integration that replaces v1 after the promotion gate passes.

## References

- [CORAL: Rank Consistent Ordinal Regression for Neural Networks](https://arxiv.org/abs/1901.07884)
- [CORN: Deep Neural Networks for Rank-Consistent Ordinal Regression](https://arxiv.org/abs/2111.08851)
- [DeepFont: Identify Your Font from an Image](https://arxiv.org/abs/1507.03196)

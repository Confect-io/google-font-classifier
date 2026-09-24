#!/usr/bin/env python3
"""Benchmark family-conditioned font-weight detection from OCR text crops.

The v6 classifier predicts a family, not a weight. This benchmark measures
whether a small image algorithm can recover the upright CSS weight after the
family is known. Calibration, confidence tuning, and final evaluation use
disjoint random seeds and text samples. The final split is the only split used
for the 95%-precision / 50%-coverage production gate.

Run a quick local check:

    uv run --with fonttools --with numpy --with onnxruntime \
      --with opencv-python-headless --with pillow --with scipy \
      --with scikit-image --with tqdm python weight_probe.py \
      --font_dir ./fonts --families 10 --per_weight 4

Run the locked benchmark with the production family model and real background
images:

    uv run --with fonttools --with numpy --with onnxruntime \
      --with opencv-python-headless --with pillow --with scipy \
      --with scikit-image --with tqdm python weight_probe.py \
      --font_dir /path/to/weight-fonts --per_weight 20 \
      --photo_dir /path/to/design-agent/eval-images \
      --model ./font-classifier-v6-deploy.onnx \
      --labels ./font_labels_v6.json --json_out /tmp/font-weight-report.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from dataset_generator import _load_text_corpus
from PIL import Image, ImageDraw, ImageFont, ImageOps

WEIGHT_NAMES = {
    "Light": 300,
    "Regular": 400,
    "Medium": 500,
    "SemiBold": 600,
    "Bold": 700,
    "ExtraBold": 800,
    "Black": 900,
}
WEIGHT_ORDER = tuple(WEIGHT_NAMES.values())
BACKGROUND_TYPES = ("flat", "gradient", "texture", "photo")
FEATURE_NAMES = (
    "skeleton_mean_stroke",
    "skeleton_median_stroke",
    "skeleton_p75_stroke",
    "ink_fill",
    "core_to_bbox_height",
)
_REFERENCE_TEXTS = (
    "Hamburgefonstiv",
    "SUMMER SALE 50%",
    "The quick brown fox",
    "299 kr.",
)
_NORM_RE = re.compile(r"[^a-z0-9]")
_MAX_CROP_WIDTH = 1600


@dataclass(frozen=True)
class FontVariant:
    family: str
    weight: int
    path: Path


@dataclass(frozen=True)
class Observation:
    features: tuple[float, ...]
    mask_agreement: float


@dataclass(frozen=True)
class FamilyCalibration:
    centroids: dict[int, tuple[float, ...]]
    scales: tuple[float, ...]


@dataclass(frozen=True)
class Prediction:
    family: str
    truth: int
    predicted: int | None
    alternative: int | None
    raw_confidence: float
    background: str
    abstention: str | None = None
    global_bold: bool | None = None
    template_predicted: int | None = None
    exact_probability: float | None = None

    @property
    def correct(self) -> bool:
        return self.predicted == self.truth

    @property
    def within_one_step(self) -> bool:
        return self.predicted is not None and abs(self.predicted - self.truth) <= 100

    @property
    def candidate_within_one_step(self) -> bool:
        return any(
            abs(weight - self.truth) <= 100
            for weight in (self.predicted, self.alternative)
            if weight is not None
        )

    @property
    def requested_group_correct(self) -> bool:
        return self.predicted is not None and _weight_group(
            self.predicted
        ) == _weight_group(self.truth)

    @property
    def candidate_group_correct(self) -> bool:
        return any(
            _weight_group(weight) == _weight_group(self.truth)
            for weight in (self.predicted, self.alternative)
            if weight is not None
        )


@dataclass(frozen=True)
class ProbabilityBin:
    lower: float
    upper: float
    exact_probability: float
    samples: int


def _weight_group(weight: int) -> str:
    if weight <= 400:
        return "300/400"
    if weight <= 600:
        return "500/600"
    return "700/800/900"


class FamilyClassifier:
    def __init__(self, model_path: Path, labels_path: Path):
        import onnxruntime as ort

        raw = json.loads(labels_path.read_text())
        self.labels = [raw[str(i)] for i in range(len(raw))]
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = int(inp.shape[-1])

    def predict(self, image: Image.Image) -> tuple[str, float]:
        from PIL import ImageOps

        rgb = image.convert("RGB")
        width, height = rgb.size
        side = max(width, height)
        pad = (
            (side - width) // 2,
            (side - height) // 2,
            side - width - (side - width) // 2,
            side - height - (side - height) // 2,
        )
        square = ImageOps.expand(rgb, pad, fill=(0, 0, 0)).resize(
            (self.input_size, self.input_size), Image.Resampling.BILINEAR
        )
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        values = (np.asarray(square, dtype=np.float32) / 255.0 - mean) / std
        values = np.transpose(values, (2, 0, 1))[np.newaxis, ...]
        logits = np.asarray(
            self.session.run(None, {self.input_name: values})[0][0],
            dtype=np.float32,
        )
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        return self.labels[index], float(probabilities[index])


def normalize_family(name: str) -> str:
    return _NORM_RE.sub("", name.lower())


def font_weight_from_metadata(path: Path) -> int:
    from fontTools.ttLib import TTFont

    font = TTFont(str(path), lazy=True)
    try:
        return int(font["OS/2"].usWeightClass)
    finally:
        font.close()


def load_font_variants(
    font_dir: Path, catalogue_csv: Path | None = None
) -> dict[str, dict[int, FontVariant]]:
    catalogue = load_catalogue_weights(catalogue_csv) if catalogue_csv else {}
    families: dict[str, dict[int, FontVariant]] = {}
    for directory in sorted(path for path in font_dir.iterdir() if path.is_dir()):
        variants: dict[int, FontVariant] = {}
        for path in sorted(directory.glob("*.ttf")):
            weight_name = path.stem.rsplit("-", 1)[-1]
            weight = WEIGHT_NAMES.get(weight_name)
            if weight is None:
                continue
            metadata_weight = font_weight_from_metadata(path)
            if metadata_weight != weight:
                raise ValueError(
                    f"{path}: filename says {weight}, OS/2 says {metadata_weight}"
                )
            variants[weight] = FontVariant(directory.name, weight, path)
        allowed = catalogue.get(normalize_family(directory.name))
        if allowed is not None:
            variants = {
                weight: item for weight, item in variants.items() if weight in allowed
            }
        if variants:
            families[directory.name] = variants
    return families


def load_catalogue_weights(path: Path) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            variants = row.get("variants", "")
            weights = {
                400 if value == "regular" else int(value)
                for value in re.findall(r"regular|(?<!italic)\b[3-9]00\b", variants)
            }
            result[normalize_family(row["name"])] = weights
    return result


def threshold_masks(image: Image.Image) -> list[np.ndarray]:
    from skimage.filters import threshold_local, threshold_otsu

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if min(gray.shape) < 8 or gray.max() - gray.min() < 8:
        return []
    global_threshold = threshold_otsu(gray)
    block_size = max(9, min(gray.shape) // 2 * 2 + 1)
    if block_size >= min(gray.shape):
        block_size = max(3, min(gray.shape) // 2 * 2 - 1)
    candidates = [gray < global_threshold, gray > global_threshold]
    if block_size >= 3:
        local_threshold = threshold_local(gray, block_size=block_size, offset=2)
        candidates.extend([gray < local_threshold, gray > local_threshold])
    return [mask for mask in candidates if _mask_is_plausible(mask)]


def _mask_is_plausible(mask: np.ndarray) -> bool:
    ink = int(mask.sum())
    fraction = ink / mask.size
    if ink < 30 or not 0.01 <= fraction <= 0.48:
        return False
    border = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    return float(border.mean()) <= 0.45


def _core_band_height(mask: np.ndarray) -> float:
    profile = mask.sum(axis=1).astype(np.float32)
    if profile.max() <= 0:
        return 0.0
    rows = np.where(profile >= 0.5 * profile.max())[0]
    return float(rows[-1] - rows[0] + 1) if len(rows) else 0.0


def _features(mask: np.ndarray) -> tuple[float, ...] | None:
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    bbox_height = float(rows[-1] - rows[0] + 1)
    bbox_width = float(cols[-1] - cols[0] + 1)
    core_height = _core_band_height(mask)
    skeleton = skeletonize(mask)
    if skeleton.sum() < 5 or core_height <= 1:
        return None
    distances = 2.0 * distance_transform_edt(mask)[skeleton]
    fill = float(mask.sum()) / (bbox_height * bbox_width)
    return (
        float(distances.mean()) / core_height,
        float(np.median(distances)) / core_height,
        float(np.percentile(distances, 75)) / core_height,
        fill,
        core_height / bbox_height,
    )


def observe(image: Image.Image) -> tuple[Observation | None, str | None]:
    measurements = [
        value for mask in threshold_masks(image) if (value := _features(mask))
    ]
    if not measurements:
        return None, "unmeasurable"
    values = np.asarray(measurements, dtype=np.float64)
    if len(values) == 1:
        agreement = 0.0
    else:
        center = np.median(values, axis=0)
        scale = np.maximum(np.abs(center), 0.02)
        distances = np.sqrt(np.mean(((values - center) / scale) ** 2, axis=1))
        agreement = float(1.0 / (1.0 + np.median(distances)))
    if agreement < 0.72:
        return None, "unstable_thresholds"
    features = tuple(float(value) for value in np.median(values, axis=0))
    return Observation(features, agreement), None


def global_bold_baseline(image: Image.Image) -> bool | None:
    import cv2

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if min(gray.shape) < 8:
        return None
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = (
        cv2.bitwise_not(thresholded)
        if np.count_nonzero(thresholded) > thresholded.size / 2
        else thresholded
    )
    rows = np.where(mask.sum(axis=1) > 0)[0]
    if np.count_nonzero(mask) < 20 or len(rows) < 2:
        return None
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 3)[mask > 0]
    if not len(distances):
        return None
    text_height = float(rows[-1] - rows[0] + 1)
    return float(np.median(distances)) * 2.0 / text_height > 0.085


def _random_color(rng: random.Random) -> tuple[int, int, int]:
    return tuple(rng.randint(0, 255) for _ in range(3))


def _background(
    kind: str,
    size: tuple[int, int],
    rng: random.Random,
    np_rng: np.random.Generator,
    photos: list[Image.Image],
) -> Image.Image:
    width, height = size
    if kind == "flat":
        return Image.new("RGB", size, _random_color(rng))
    if kind == "gradient":
        start = np.asarray(_random_color(rng), dtype=np.float32)
        end = np.asarray(_random_color(rng), dtype=np.float32)
        amount = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
        values = start + (end - start) * amount
        values = np.repeat(values, height, axis=0)
        return Image.fromarray(values.astype(np.uint8), "RGB")
    if kind == "photo" and photos:
        photo = photos[rng.randrange(len(photos))]
        return ImageOps.fit(
            photo,
            size,
            Image.Resampling.LANCZOS,
            centering=(rng.random(), rng.random()),
        )
    coarse = np_rng.integers(
        0,
        256,
        size=(max(2, height // 12), max(2, width // 12), 3),
        dtype=np.uint8,
    )
    texture = Image.fromarray(coarse, "RGB").resize(size, Image.Resampling.BICUBIC)
    fine = np_rng.normal(0, 18, size=(height, width, 3))
    values = np.asarray(texture, dtype=np.float32) + fine
    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8), "RGB")


def render_crop(
    variant: FontVariant,
    text: str,
    background: str,
    rng: random.Random,
    np_rng: np.random.Generator,
    photos: list[Image.Image],
) -> Image.Image:
    source_size = rng.randint(90, 220)
    target_height = rng.randint(16, 150)
    font = ImageFont.truetype(
        str(variant.path), source_size, layout_engine=ImageFont.Layout.BASIC
    )
    bbox = font.getbbox(text)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(4, height // 5)
    while (width + 2 * pad) * target_height / (
        height + 2 * pad
    ) > _MAX_CROP_WIDTH and len(text) > 4:
        text = text[:-1]
        bbox = font.getbbox(text)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = max(4, height // 5)
    canvas_size = (max(8, width + 2 * pad), max(8, height + 2 * pad))
    canvas = _background(background, canvas_size, rng, np_rng, photos)
    mean = np.asarray(canvas, dtype=np.float32).mean(axis=(0, 1))
    foreground = (0, 0, 0) if mean.mean() > 130 else (255, 255, 255)
    ImageDraw.Draw(canvas).text(
        (pad - bbox[0], pad - bbox[1]), text, font=font, fill=foreground
    )
    target_width = max(4, round(canvas.width * target_height / canvas.height))
    canvas = canvas.resize((target_width, target_height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    canvas.save(buffer, "JPEG", quality=rng.randint(55, 95), subsampling=2)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def load_photos(photo_dir: Path | None) -> list[Image.Image]:
    if photo_dir is None:
        return []
    paths = [
        path
        for path in sorted(photo_dir.rglob("*"))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    return [Image.open(path).convert("RGB") for path in paths]


def choose_text(corpus: list[str], rng: random.Random) -> str:
    content = rng.choice(corpus)
    length = rng.randint(8, 38)
    start = rng.randint(0, max(0, len(content) - length))
    text = content[start : start + length].strip().replace("\n", " ")
    if rng.random() < 0.18:
        text = text.upper()
    return text or _REFERENCE_TEXTS[0]


def build_calibrations(
    families: dict[str, dict[int, FontVariant]],
    samples: int,
    seed: int,
    photos: list[Image.Image],
) -> dict[str, FamilyCalibration]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    calibrations: dict[str, FamilyCalibration] = {}
    for index, (family, variants) in enumerate(families.items(), start=1):
        values: dict[int, list[tuple[float, ...]]] = defaultdict(list)
        for weight, variant in variants.items():
            attempts = 0
            while len(values[weight]) < samples and attempts < samples * 12:
                attempts += 1
                text = _REFERENCE_TEXTS[attempts % len(_REFERENCE_TEXTS)]
                image = render_crop(variant, text, "flat", rng, np_rng, photos)
                observation, _ = observe(image)
                if observation:
                    values[weight].append(observation.features)
        if any(not values[weight] for weight in variants):
            continue
        centroids = {
            weight: tuple(float(value) for value in np.median(items, axis=0))
            for weight, items in values.items()
        }
        all_values = np.asarray([item for items in values.values() for item in items])
        scales = tuple(float(max(value, 0.02)) for value in np.std(all_values, axis=0))
        calibrations[family] = FamilyCalibration(centroids, scales)
        if index % 20 == 0 or index == len(families):
            print(f"calibrated {index}/{len(families)} families", flush=True)
    return calibrations


def predict_weight(
    family: str,
    observation: Observation,
    calibrations: dict[str, FamilyCalibration],
) -> tuple[int, int | None, float] | None:
    calibration = calibrations.get(family)
    if calibration is None:
        return None
    target = np.asarray(observation.features)
    scale = np.asarray(calibration.scales)
    distances = sorted(
        (
            float(np.sqrt(np.mean(((target - np.asarray(features)) / scale) ** 2))),
            weight,
        )
        for weight, features in calibration.centroids.items()
    )
    if len(distances) == 1:
        return distances[0][1], None, 0.0
    first, second = distances[:2]
    ordered_weights = sorted(calibration.centroids)
    predicted_index = ordered_weights.index(first[1])
    adjacent_weights = ordered_weights[
        max(0, predicted_index - 1) : predicted_index + 2
    ]
    adjacent_weights.remove(first[1])
    alternative = min(
        (item for item in distances if item[1] in adjacent_weights),
        default=second,
    )[1]
    confidence = second[0] / (first[0] + second[0] + 1e-12)
    return first[1], alternative, float(confidence * observation.mask_agreement)


def same_text_prediction(
    family: str,
    text: str,
    background: str,
    seed: int,
    target: Observation,
    families: dict[str, dict[int, FontVariant]],
    photos: list[Image.Image],
) -> int | None:
    variants = families.get(family)
    if not variants:
        return None
    candidates: list[tuple[float, int]] = []
    scale = np.maximum(np.abs(np.asarray(target.features)), 0.02)
    for weight, variant in variants.items():
        image = render_crop(
            variant,
            text,
            background,
            random.Random(seed),
            np.random.default_rng(seed),
            photos,
        )
        observation, _ = observe(image)
        if observation is None:
            continue
        distance = float(
            np.sqrt(
                np.mean(
                    ((np.asarray(observation.features) - target.features) / scale) ** 2
                )
            )
        )
        candidates.append((distance, weight))
    return min(candidates)[1] if candidates else None


def _classifier_family(
    label: str, families: dict[str, dict[int, FontVariant]]
) -> str | None:
    by_normalized = {normalize_family(family): family for family in families}
    return by_normalized.get(normalize_family(label))


def evaluate_split(
    families: dict[str, dict[int, FontVariant]],
    calibrations: dict[str, FamilyCalibration],
    per_weight: int,
    seed: int,
    photos: list[Image.Image],
    classifier: FamilyClassifier | None,
    include_same_text: bool,
) -> list[Prediction]:
    rng = random.Random(seed)
    corpus = _load_text_corpus()
    predictions: list[Prediction] = []
    for family_index, (family, variants) in enumerate(families.items(), start=1):
        if len(variants) < 2:
            continue
        for truth, variant in variants.items():
            for index in range(per_weight):
                background = BACKGROUND_TYPES[index % len(BACKGROUND_TYPES)]
                text = choose_text(corpus, rng)
                render_seed = rng.randrange(2**32)
                image = render_crop(
                    variant,
                    text,
                    background,
                    random.Random(render_seed),
                    np.random.default_rng(render_seed),
                    photos,
                )
                global_bold = global_bold_baseline(image)
                observation, reason = observe(image)
                predicted_family = family
                if classifier is not None:
                    label, _ = classifier.predict(image)
                    predicted_family = _classifier_family(label, families) or ""
                if observation is None:
                    predictions.append(
                        Prediction(
                            family,
                            truth,
                            None,
                            None,
                            0.0,
                            background,
                            reason,
                            global_bold,
                        )
                    )
                    continue
                result = predict_weight(predicted_family, observation, calibrations)
                if result is None:
                    predictions.append(
                        Prediction(
                            family,
                            truth,
                            None,
                            None,
                            0.0,
                            background,
                            "family_unavailable",
                            global_bold,
                        )
                    )
                    continue
                predicted, alternative, confidence = result
                template_predicted = (
                    same_text_prediction(
                        predicted_family,
                        text,
                        background,
                        render_seed,
                        observation,
                        families,
                        photos,
                    )
                    if include_same_text
                    else None
                )
                predictions.append(
                    Prediction(
                        family,
                        truth,
                        predicted,
                        alternative,
                        confidence,
                        background,
                        global_bold=global_bold,
                        template_predicted=template_predicted,
                    )
                )
        if family_index % 20 == 0 or family_index == len(families):
            print(
                f"evaluated {family_index}/{len(families)} families",
                flush=True,
            )
    return predictions


def calibrate_exact_probabilities(
    predictions: list[Prediction], bin_count: int = 10
) -> tuple[ProbabilityBin, ...]:
    return calibrate_probabilities(predictions, lambda item: item.correct, bin_count)


def calibrate_probabilities(
    predictions: list[Prediction],
    success: Callable[[Prediction], bool],
    bin_count: int = 10,
) -> tuple[ProbabilityBin, ...]:
    measurable = sorted(
        (item for item in predictions if item.predicted is not None),
        key=lambda item: item.raw_confidence,
    )
    if not measurable:
        return ()
    target_size = max(1, math.ceil(len(measurable) / bin_count))
    raw_bins: list[list[Prediction]] = []
    current: list[Prediction] = []
    for index, item in enumerate(measurable):
        current.append(item)
        next_confidence = (
            measurable[index + 1].raw_confidence
            if index + 1 < len(measurable)
            else None
        )
        if len(current) >= target_size and next_confidence != item.raw_confidence:
            raw_bins.append(current)
            current = []
    if current:
        raw_bins.append(current)

    blocks = [
        {
            "lower": items[0].raw_confidence,
            "upper": items[-1].raw_confidence,
            "correct": sum(success(item) for item in items),
            "samples": len(items),
        }
        for items in raw_bins
    ]
    index = 1
    while index < len(blocks):
        previous = blocks[index - 1]
        current_block = blocks[index]
        previous_rate = previous["correct"] / previous["samples"]
        current_rate = current_block["correct"] / current_block["samples"]
        if previous_rate <= current_rate:
            index += 1
            continue
        previous["upper"] = current_block["upper"]
        previous["correct"] += current_block["correct"]
        previous["samples"] += current_block["samples"]
        blocks.pop(index)
        index = max(1, index - 1)
    return tuple(
        ProbabilityBin(
            lower=float(block["lower"]),
            upper=float(block["upper"]),
            exact_probability=float(block["correct"] / block["samples"]),
            samples=int(block["samples"]),
        )
        for block in blocks
    )


def exact_probability(
    raw_confidence: float, calibration: tuple[ProbabilityBin, ...]
) -> float | None:
    if not calibration:
        return None
    for item in calibration:
        if raw_confidence <= item.upper:
            return item.exact_probability
    return calibration[-1].exact_probability


def apply_probability_calibration(
    predictions: list[Prediction], calibration: tuple[ProbabilityBin, ...]
) -> list[Prediction]:
    return [
        replace(
            item,
            exact_probability=(
                exact_probability(item.raw_confidence, calibration)
                if item.predicted is not None
                else None
            ),
        )
        for item in predictions
    ]


def _selection_confidence(prediction: Prediction) -> float:
    return (
        prediction.exact_probability
        if prediction.exact_probability is not None
        else prediction.raw_confidence
    )


def select_authority_threshold(
    predictions: list[Prediction], precision_target: float
) -> float | None:
    return select_threshold(predictions, precision_target, lambda item: item.correct)


def select_threshold(
    predictions: list[Prediction],
    precision_target: float,
    success: Callable[[Prediction], bool],
) -> float | None:
    measurable = [item for item in predictions if item.predicted is not None]
    candidates = sorted({_selection_confidence(item) for item in measurable})
    selected = None
    selected_coverage = -1.0
    for threshold in candidates:
        accepted = [
            item for item in measurable if _selection_confidence(item) >= threshold
        ]
        precision = sum(success(item) for item in accepted) / len(accepted)
        coverage = len(accepted) / len(predictions)
        if precision >= precision_target and coverage > selected_coverage:
            selected = threshold
            selected_coverage = coverage
    return selected


def select_raw_threshold(
    predictions: list[Prediction],
    precision_target: float,
    success: Callable[[Prediction], bool],
) -> float | None:
    measurable = [item for item in predictions if item.predicted is not None]
    candidates = sorted({item.raw_confidence for item in measurable})
    selected = None
    selected_coverage = -1.0
    for threshold in candidates:
        accepted = [item for item in measurable if item.raw_confidence >= threshold]
        precision = sum(success(item) for item in accepted) / len(accepted)
        coverage = len(accepted) / len(predictions)
        if precision >= precision_target and coverage > selected_coverage:
            selected = threshold
            selected_coverage = coverage
    return selected


def export_calibration(
    path: Path,
    families: dict[str, dict[int, FontVariant]],
    calibrations: dict[str, FamilyCalibration],
    probability_calibration: tuple[ProbabilityBin, ...],
    advisory_threshold: float | None,
    advisory_precision: float,
) -> None:
    payload = {
        "schema_version": 1,
        "features": FEATURE_NAMES,
        "advisory_metric": "top_two_requested_weight_group",
        "requested_weight_groups": [[300, 400], [500, 600], [700, 800, 900]],
        "advisory_precision_target": advisory_precision,
        "advisory_raw_confidence_threshold": (
            round(advisory_threshold, 6) if advisory_threshold is not None else None
        ),
        "probability_calibration": [
            {
                "raw_confidence_upper": round(item.upper, 6),
                "probability": round(item.exact_probability, 6),
                "samples": item.samples,
            }
            for item in probability_calibration
        ],
        "families": {
            family: {
                "supported_weights": sorted(families[family]),
                "centroids": {
                    str(weight): [round(value, 6) for value in features]
                    for weight, features in calibration.centroids.items()
                },
                "scales": [round(value, 6) for value in calibration.scales],
            }
            for family, calibration in sorted(calibrations.items())
        },
    }
    path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    )


def summarize(
    predictions: list[Prediction],
    threshold: float | None,
    categories: dict[str, str],
) -> dict:
    measurable = [item for item in predictions if item.predicted is not None]
    accepted = (
        [item for item in measurable if _selection_confidence(item) >= threshold]
        if threshold is not None
        else []
    )
    confusion = Counter((item.truth, item.predicted) for item in measurable)
    abstentions = Counter(item.abstention for item in predictions if item.abstention)

    def accuracy(items: list[Prediction]) -> float:
        return sum(item.correct for item in items) / len(items) if items else 0.0

    global_rows = [item for item in predictions if item.global_bold is not None]
    global_emitted = [item for item in global_rows if item.global_bold]
    template_rows = [
        item for item in predictions if item.template_predicted is not None
    ]

    by_background = {
        background: {
            "samples": len(items),
            "measurable": sum(item.predicted is not None for item in items),
            "accuracy": accuracy(
                [item for item in items if item.predicted is not None]
            ),
        }
        for background in BACKGROUND_TYPES
        if (items := [item for item in predictions if item.background == background])
    }
    by_family = {
        family: {
            "samples": len(items),
            "coverage": sum(item.predicted is not None for item in items) / len(items),
            "accuracy": accuracy(
                [item for item in items if item.predicted is not None]
            ),
        }
        for family in sorted({item.family for item in predictions})
        if (items := [item for item in predictions if item.family == family])
    }
    by_category = {
        category: {
            "samples": len(items),
            "coverage": sum(item.predicted is not None for item in items) / len(items),
            "accuracy": accuracy(
                [item for item in items if item.predicted is not None]
            ),
        }
        for category in sorted(set(categories.values()))
        if (
            items := [
                item for item in predictions if categories.get(item.family) == category
            ]
        )
    }
    precision_curve = []
    for candidate in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        items = [
            item for item in measurable if _selection_confidence(item) >= candidate
        ]
        precision_curve.append(
            {
                "threshold": candidate,
                "precision": accuracy(items),
                "coverage": len(items) / len(predictions),
            }
        )
    return {
        "samples": len(predictions),
        "measurable_coverage": len(measurable) / len(predictions),
        "measurable_accuracy": accuracy(measurable),
        "within_one_step_accuracy": (
            sum(item.within_one_step for item in measurable) / len(measurable)
            if measurable
            else 0.0
        ),
        "requested_group_accuracy": (
            sum(item.requested_group_correct for item in measurable) / len(measurable)
            if measurable
            else 0.0
        ),
        "candidate_within_one_step": (
            sum(item.candidate_within_one_step for item in measurable) / len(measurable)
            if measurable
            else 0.0
        ),
        "candidate_group_accuracy": (
            sum(item.candidate_group_correct for item in measurable) / len(measurable)
            if measurable
            else 0.0
        ),
        "authority_threshold": threshold,
        "authoritative_precision": accuracy(accepted),
        "authoritative_coverage": len(accepted) / len(predictions),
        "global_bold_accuracy": (
            sum(item.global_bold == (item.truth >= 600) for item in global_rows)
            / len(global_rows)
            if global_rows
            else 0.0
        ),
        "global_bold_emitted_precision": (
            sum(item.truth >= 600 for item in global_emitted) / len(global_emitted)
            if global_emitted
            else 0.0
        ),
        "same_text_accuracy": (
            sum(item.template_predicted == item.truth for item in template_rows)
            / len(template_rows)
            if template_rows
            else None
        ),
        "confusion": {
            f"{truth}->{predicted}": count
            for (truth, predicted), count in sorted(confusion.items())
        },
        "abstentions": dict(sorted(abstentions.items())),
        "by_background": by_background,
        "by_family": by_family,
        "by_category": by_category,
        "precision_curve": precision_curve,
    }


def serialize_predictions(predictions: list[Prediction]) -> list[dict]:
    return [
        {
            "family": item.family,
            "truth": item.truth,
            "ranked_weights": [
                weight
                for weight in (item.predicted, item.alternative)
                if weight is not None
            ],
            "exact_probability": item.exact_probability,
            "raw_confidence": item.raw_confidence,
            "background": item.background,
            "abstention": item.abstention,
        }
        for item in predictions
    ]


def _print_summary(name: str, summary: dict) -> None:
    print(f"\n{name}")
    print(f"  samples:                 {summary['samples']}")
    print(f"  measurable coverage:     {summary['measurable_coverage']:.1%}")
    print(f"  measurable exact:        {summary['measurable_accuracy']:.1%}")
    print(f"  top-1 within 100:         {summary['within_one_step_accuracy']:.1%}")
    print(f"  top-1 requested group:    {summary['requested_group_accuracy']:.1%}")
    print(f"  top-2 within 100:         {summary['candidate_within_one_step']:.1%}")
    print(f"  top-2 requested group:    {summary['candidate_group_accuracy']:.1%}")
    print(f"  authoritative precision: {summary['authoritative_precision']:.1%}")
    print(f"  authoritative coverage:  {summary['authoritative_coverage']:.1%}")
    print(f"  global bold accuracy:     {summary['global_bold_accuracy']:.1%}")
    print(f"  emitted bold precision:  {summary['global_bold_emitted_precision']:.1%}")
    if summary["same_text_accuracy"] is not None:
        print(f"  same-text upper bound:    {summary['same_text_accuracy']:.1%}")
    for background, values in summary["by_background"].items():
        print(
            f"    {background:<9} accuracy {values['accuracy']:.1%}, "
            f"measurable {values['measurable']}/{values['samples']}"
        )
    if summary["abstentions"]:
        print(f"  abstentions: {summary['abstentions']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font_dir", type=Path, default=Path("./fonts"))
    parser.add_argument("--photo_dir", type=Path)
    parser.add_argument("--catalogue_csv", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--labels", type=Path, default=Path("./font_labels_v6.json"))
    parser.add_argument("--families", type=int)
    parser.add_argument("--calibration_samples", type=int, default=12)
    parser.add_argument("--validation_per_weight", type=int, default=12)
    parser.add_argument("--per_weight", type=int, default=20)
    parser.add_argument("--precision_gate", type=float, default=0.95)
    parser.add_argument("--coverage_gate", type=float, default=0.50)
    parser.add_argument("--advisory_precision", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--json_out", type=Path)
    parser.add_argument("--calibration_out", type=Path)
    parser.add_argument("--skip_final", action="store_true")
    parser.add_argument("--skip_same_text", action="store_true")
    parser.add_argument(
        "--categories", type=Path, default=Path("./font_categories_v6.json")
    )
    args = parser.parse_args()

    families = load_font_variants(args.font_dir, args.catalogue_csv)
    if args.families:
        selected = random.Random(args.seed).sample(
            sorted(families), min(args.families, len(families))
        )
        families = {family: families[family] for family in selected}
    multi_weight = sum(len(variants) >= 2 for variants in families.values())
    print(
        f"loaded {len(families)} families ({multi_weight} multi-weight), "
        f"{sum(len(items) for items in families.values())} upright variants"
    )
    photos = load_photos(args.photo_dir)
    raw_categories = (
        json.loads(args.categories.read_text()) if args.categories.exists() else {}
    )
    categories = {family: raw_categories.get(family, "unknown") for family in families}
    classifier = (
        FamilyClassifier(args.model, args.labels) if args.model is not None else None
    )
    calibrations = build_calibrations(
        families, args.calibration_samples, args.seed, photos
    )
    validation = evaluate_split(
        families,
        calibrations,
        args.validation_per_weight,
        args.seed + 1,
        photos,
        classifier,
        not args.skip_same_text,
    )
    probability_calibration = calibrate_exact_probabilities(validation)
    validation = apply_probability_calibration(validation, probability_calibration)
    threshold = select_authority_threshold(validation, args.precision_gate)
    advisory_probability_calibration = calibrate_probabilities(
        validation, lambda item: item.candidate_group_correct
    )
    advisory_threshold = select_raw_threshold(
        validation,
        args.advisory_precision,
        lambda item: item.candidate_group_correct,
    )
    if args.calibration_out:
        export_calibration(
            args.calibration_out,
            families,
            calibrations,
            advisory_probability_calibration,
            advisory_threshold,
            args.advisory_precision,
        )
    if args.skip_final:
        validation_summary = summarize(validation, threshold, categories)
        _print_summary("validation", validation_summary)
        return 0
    final = evaluate_split(
        families,
        calibrations,
        args.per_weight,
        args.seed + 2,
        photos,
        classifier,
        not args.skip_same_text,
    )
    final = apply_probability_calibration(final, probability_calibration)
    validation_summary = summarize(validation, threshold, categories)
    final_summary = summarize(final, threshold, categories)
    _print_summary("validation", validation_summary)
    _print_summary("locked final", final_summary)
    promoted = (
        final_summary["authoritative_precision"] >= args.precision_gate
        and final_summary["authoritative_coverage"] >= args.coverage_gate
    )
    print(
        f"\nproduction gate: {'PASS' if promoted else 'FAIL'} "
        f"(need precision >= {args.precision_gate:.0%}, "
        f"coverage >= {args.coverage_gate:.0%})"
    )
    report = {
        "schema_version": 1,
        "seed": args.seed,
        "features": FEATURE_NAMES,
        "families": len(families),
        "multi_weight_families": multi_weight,
        "family_classifier": classifier is not None,
        "probability_calibration": [
            {
                "raw_confidence_lower": item.lower,
                "raw_confidence_upper": item.upper,
                "exact_probability": item.exact_probability,
                "samples": item.samples,
            }
            for item in probability_calibration
        ],
        "validation": validation_summary,
        "final": final_summary,
        "final_predictions": serialize_predictions(final),
        "production_gate": {
            "precision": args.precision_gate,
            "coverage": args.coverage_gate,
            "passed": promoted,
        },
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())

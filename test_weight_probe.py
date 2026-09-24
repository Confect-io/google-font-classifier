from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import weight_probe
from PIL import Image, ImageDraw
from weight_probe import (
    FamilyCalibration,
    FontVariant,
    Observation,
    Prediction,
    ProbabilityBin,
    apply_probability_calibration,
    build_calibrations,
    calibrate_exact_probabilities,
    exact_probability,
    load_font_variants,
    observe,
    predict_weight,
    select_authority_threshold,
    threshold_masks,
)


def text_crop(background, foreground):
    image = Image.new("RGB", (220, 64), background)
    ImageDraw.Draw(image).text((15, 12), "Weight 700", fill=foreground)
    return image


def test_threshold_masks_support_both_polarities():
    assert threshold_masks(text_crop("white", "black"))
    assert threshold_masks(text_crop("black", "white"))


def test_tiny_and_flat_crops_abstain():
    observation, reason = observe(Image.new("RGB", (5, 5), "white"))
    assert observation is None
    assert reason == "unmeasurable"

    observation, reason = observe(Image.new("RGB", (100, 30), "gray"))
    assert observation is None
    assert reason == "unmeasurable"


def test_observation_is_deterministic():
    image = text_crop("white", "black")
    first = observe(image)
    second = observe(image)
    assert first == second


def test_authority_threshold_uses_precision_not_raw_margin():
    rows = [
        Prediction("A", 400, 400, 500, 0.9, "flat"),
        Prediction("A", 400, 400, 500, 0.8, "flat"),
        Prediction("A", 400, 500, 400, 0.7, "flat"),
        Prediction("A", 400, None, None, 0.0, "flat", "unmeasurable"),
    ]
    assert select_authority_threshold(rows, 1.0) == 0.8


def test_relaxed_weight_metrics_use_requested_groups_and_ranked_candidates():
    prediction = Prediction("A", 600, 700, 500, 0.8, "flat")

    assert prediction.within_one_step
    assert prediction.candidate_within_one_step
    assert not prediction.requested_group_correct
    assert prediction.candidate_group_correct


def test_exact_probability_is_empirically_calibrated_and_lookup_is_bounded():
    rows = [
        Prediction("A", 400, 500, 400, 0.1, "flat"),
        Prediction("A", 400, 400, 500, 0.2, "flat"),
        Prediction("A", 400, 400, 500, 0.8, "flat"),
        Prediction("A", 400, 400, 500, 0.9, "flat"),
    ]

    calibration = calibrate_exact_probabilities(rows, bin_count=2)
    calibrated = apply_probability_calibration(rows, calibration)

    assert calibration == (
        ProbabilityBin(0.1, 0.2, 0.5, 2),
        ProbabilityBin(0.8, 0.9, 1.0, 2),
    )
    assert exact_probability(0.0, calibration) == 0.5
    assert exact_probability(1.0, calibration) == 1.0
    assert [item.exact_probability for item in calibrated] == [0.5, 0.5, 1.0, 1.0]


def test_textured_noise_is_rejected_or_low_agreement():
    rng = np.random.default_rng(3)
    image = Image.fromarray(rng.integers(0, 256, (80, 240, 3), dtype=np.uint8))
    observation, _ = observe(image)
    assert observation is None or observation.mask_agreement < 0.8


def test_font_metadata_must_match_filename(tmp_path, monkeypatch):
    family_dir = tmp_path / "Example"
    family_dir.mkdir()
    (family_dir / "Example-Bold.ttf").touch()
    monkeypatch.setattr(weight_probe, "font_weight_from_metadata", lambda _: 400)

    with pytest.raises(ValueError, match="filename says 700, OS/2 says 400"):
        load_font_variants(tmp_path)


def test_catalogue_filters_unsupported_weights(tmp_path, monkeypatch):
    family_dir = tmp_path / "Example"
    family_dir.mkdir()
    for name in ("Regular", "Bold", "Black"):
        (family_dir / f"Example-{name}.ttf").touch()
    weights = {"Regular": 400, "Bold": 700, "Black": 900}
    monkeypatch.setattr(
        weight_probe,
        "font_weight_from_metadata",
        lambda path: weights[path.stem.rsplit("-", 1)[-1]],
    )
    catalogue = tmp_path / "fonts.csv"
    catalogue.write_text('name,variants\nExample,"regular,700"\n')

    families = load_font_variants(tmp_path, catalogue)

    assert set(families["Example"]) == {400, 700}


def test_prediction_never_leaves_supported_weights():
    calibration = FamilyCalibration(
        centroids={300: (0.1,) * 5, 700: (0.9,) * 5},
        scales=(0.1,) * 5,
    )
    observation = Observation((0.8,) * 5, 0.9)

    predicted, alternative, _ = predict_weight(
        "Example", observation, {"Example": calibration}
    )

    assert predicted in calibration.centroids
    assert alternative in calibration.centroids


def test_single_weight_family_returns_only_variant_without_confidence():
    calibration = FamilyCalibration(
        centroids={400: (0.2,) * 5},
        scales=(0.1,) * 5,
    )
    observation = Observation((0.8,) * 5, 0.9)

    assert predict_weight("Example", observation, {"Example": calibration}) == (
        400,
        None,
        0.0,
    )


def test_calibration_generation_is_deterministic(monkeypatch):
    families = {
        "Example": {
            400: FontVariant("Example", 400, Path("Example-Regular.ttf")),
            700: FontVariant("Example", 700, Path("Example-Bold.ttf")),
        }
    }
    monkeypatch.setattr(
        weight_probe,
        "render_crop",
        lambda variant, *_: Image.new("RGB", (variant.weight // 100, 10)),
    )
    monkeypatch.setattr(
        weight_probe,
        "observe",
        lambda image: (
            Observation((float(image.width),) * 5, 1.0),
            None,
        ),
    )

    first = build_calibrations(families, 3, 123, [])
    second = build_calibrations(families, 3, 123, [])

    assert first == second

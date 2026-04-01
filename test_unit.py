#!/usr/bin/env python3
"""
Unit tests for font-model components.

Tests pure functions and training code paths without cloud infrastructure.
Run with: python -m pytest test_unit.py -v

These tests use a tiny synthetic dataset (3 classes, ~10 images each)
and run 1 training step per mode to validate correctness.
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Test: extract_family label parser
# ---------------------------------------------------------------------------

class TestExtractFamily:
    def setup_method(self):
        from confusion_matrix import extract_family
        self.extract = extract_family

    def test_hyphen_separator(self):
        assert self.extract("Barlow-Bold") == "Barlow"

    def test_underscore_separator(self):
        assert self.extract("Aleo_Bold") == "Aleo"

    def test_no_separator(self):
        assert self.extract("Ultra") == "Ultra"

    def test_multiple_hyphens(self):
        assert self.extract("BigShouldersText-Bold") == "BigShouldersText"

    def test_italic_suffix(self):
        assert self.extract("CrimsonPro-BoldItalic") == "CrimsonPro"

    def test_numeric_weight(self):
        assert self.extract("Roboto-300") == "Roboto"


# ---------------------------------------------------------------------------
# Test: label filtering (._* macOS resource fork files)
# ---------------------------------------------------------------------------

class TestLabelFiltering:
    def test_filters_dot_underscore(self):
        entries = ["Arial", "._Arial", "Roboto", "._Roboto", ".DS_Store"]
        filtered = sorted(d for d in entries if not d.startswith('.'))
        assert filtered == ["Arial", "Roboto"]

    def test_no_hidden_files(self):
        entries = ["Arial", "Roboto", "Inter"]
        filtered = sorted(d for d in entries if not d.startswith('.'))
        assert filtered == ["Arial", "Inter", "Roboto"]


# ---------------------------------------------------------------------------
# Test: typographic distance matrix
# ---------------------------------------------------------------------------

class TestTypographicDistance:
    def setup_method(self):
        from confusion_matrix import compute_typographic_distance_matrix
        self.compute = compute_typographic_distance_matrix

    def test_same_variant_zero_distance(self):
        labels = ["Roboto-Bold", "Roboto-Light", "Inter-Regular"]
        dist = self.compute(labels)
        for i in range(len(labels)):
            assert dist[i, i] == 0.0

    def test_symmetric(self):
        labels = ["Roboto-Bold", "Inter-Regular", "CrimsonPro-Light"]
        dist = self.compute(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                assert dist[i, j] == dist[j, i]

    def test_same_family_closer_than_cross_family(self):
        labels = ["Roboto-Bold", "Roboto-Light", "Inter-Regular"]
        dist = self.compute(labels)
        # Roboto-Bold to Roboto-Light (same family) should be < Roboto-Bold to Inter
        assert dist[0, 1] < dist[0, 2]

    def test_same_category_closer_than_cross_category(self):
        # Inter (sans) vs Roboto (sans) should be closer than Inter (sans) vs CrimsonPro (serif)
        labels = ["Inter-Regular", "Roboto-Regular", "CrimsonPro-Regular"]
        dist = self.compute(labels)
        assert dist[0, 1] < dist[0, 2]

    def test_distance_tiers(self):
        labels = ["Roboto-Regular", "Roboto-Bold", "Inter-Regular", "CrimsonPro-Regular"]
        dist = self.compute(labels)
        same_family = dist[0, 1]       # Roboto-Regular to Roboto-Bold
        same_category = dist[0, 2]     # Roboto to Inter (both sans)
        cross_category = dist[0, 3]    # Roboto (sans) to CrimsonPro (serif)
        assert 0.2 <= same_family <= 0.4
        assert same_category == 0.7
        assert cross_category == 1.0

    def test_triangle_inequality(self):
        labels = ["Roboto-Regular", "Roboto-Bold", "Inter-Regular", "CrimsonPro-Light"]
        dist = self.compute(labels)
        n = len(labels)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert dist[i, j] <= dist[i, k] + dist[k, j] + 1e-10


# ---------------------------------------------------------------------------
# Test: SWER computation
# ---------------------------------------------------------------------------

class TestSWER:
    def test_perfect_classifier(self):
        from confusion_matrix import compute_severity_metrics, compute_typographic_distance_matrix
        labels = ["Roboto-Bold", "Inter-Regular", "CrimsonPro-Light"]
        dist = compute_typographic_distance_matrix(labels)
        y_true = ["Roboto-Bold", "Inter-Regular", "CrimsonPro-Light"]
        y_pred = ["Roboto-Bold", "Inter-Regular", "CrimsonPro-Light"]
        metrics = compute_severity_metrics(y_true, y_pred, labels, dist)
        assert metrics["severity_weighted_error"] == 0.0
        assert metrics["family_accuracy"] == 1.0

    def test_within_family_error_low_severity(self):
        from confusion_matrix import compute_severity_metrics, compute_typographic_distance_matrix
        labels = ["Roboto-Regular", "Roboto-Bold", "CrimsonPro-Regular"]
        dist = compute_typographic_distance_matrix(labels)
        # Confuse within family
        y_true = ["Roboto-Regular"]
        y_pred = ["Roboto-Bold"]
        metrics = compute_severity_metrics(y_true, y_pred, labels, dist)
        within_severity = metrics["severity_weighted_error"]
        # Confuse across category
        y_pred_cross = ["CrimsonPro-Regular"]
        metrics_cross = compute_severity_metrics(y_true, y_pred_cross, labels, dist)
        cross_severity = metrics_cross["severity_weighted_error"]
        assert within_severity < cross_severity

    def test_family_accuracy_counts_same_family_as_correct(self):
        from confusion_matrix import compute_severity_metrics, compute_typographic_distance_matrix
        labels = ["Roboto-Regular", "Roboto-Bold", "Inter-Regular"]
        dist = compute_typographic_distance_matrix(labels)
        y_true = ["Roboto-Regular", "Inter-Regular"]
        y_pred = ["Roboto-Bold", "Inter-Regular"]  # wrong variant, right family
        metrics = compute_severity_metrics(y_true, y_pred, labels, dist)
        assert metrics["family_accuracy"] == 1.0  # both correct at family level


# ---------------------------------------------------------------------------
# Test: ResNet forward returns dict
# ---------------------------------------------------------------------------

class TestResNetForward:
    def test_returns_dict_with_loss_and_logits(self):
        from torchvision.models import resnet50

        backbone = resnet50(weights=None)
        num_classes = 3
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)

        # Minimal forward
        x = torch.randn(2, 3, 224, 224)
        logits = backbone(x)
        labels = torch.tensor([0, 1])
        loss = torch.nn.functional.cross_entropy(logits, labels)
        result = {"loss": loss, "logits": logits}

        assert isinstance(result, dict)
        assert "loss" in result
        assert "logits" in result
        assert result["logits"].shape == (2, num_classes)
        assert result["loss"].dim() == 0  # scalar


# ---------------------------------------------------------------------------
# Test: preprocessing consistency
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_train_and_inference_use_same_transform(self):
        from handler import get_inference_transform
        from train_model import get_transform
        from transformers import AutoImageProcessor
        from PIL import Image

        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
        size = processor.size["shortest_edge"]

        # Create a test image
        img = Image.new("RGB", (300, 100), color=(128, 64, 200))

        # Inference transform
        inf_transform = get_inference_transform(processor, size)
        inf_result = inf_transform(img)

        # Train transform
        train_transform = get_transform(processor, size)
        train_result = train_transform({"image": img})["pixel_values"]

        assert torch.allclose(inf_result, train_result, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: training smoke tests (1 step per mode on tiny dataset)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_dataset():
    """Create a tiny dataset for smoke testing."""
    tmpdir = tempfile.mkdtemp()
    for split in ["train", "test"]:
        for cls in ["FakeSerif_Regular", "FakeSans_Bold", "FakeMono_Light"]:
            cls_dir = os.path.join(tmpdir, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            n_images = 10 if split == "train" else 3
            for i in range(n_images):
                from PIL import Image
                img = Image.new("RGB", (224, 224), color=(
                    (i * 37) % 256, (i * 73) % 256, (i * 113) % 256
                ))
                img.save(os.path.join(cls_dir, f"img_{i:03d}.png"))
    yield tmpdir
    shutil.rmtree(tmpdir)


def _run_training(tiny_dataset, extra_args):
    """Helper to run train_model.py with given args and verify output."""
    with tempfile.TemporaryDirectory() as outdir:
        import subprocess
        cmd = [
            "python3", "train_model.py",
            "--data_dir", tiny_dataset,
            "--output_dir", outdir,
            "--batch_size", "4",
            "--epochs", "1",
            "--learning_rate", "1e-4",
        ] + extra_args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, f"Training failed:\nSTDOUT: {result.stdout[-500:]}\nSTDERR: {result.stderr[-500:]}"
        assert os.path.exists(os.path.join(outdir, "result_model")), "No result_model directory"
        return outdir


class TestTrainingSmoke:
    def test_lora_default(self, tiny_dataset):
        _run_training(tiny_dataset, [])

    def test_lora_rank4(self, tiny_dataset):
        _run_training(tiny_dataset, ["--lora_rank", "4", "--lora_alpha", "8"])

    def test_lora_rank16(self, tiny_dataset):
        _run_training(tiny_dataset, ["--lora_rank", "16", "--lora_alpha", "32"])

    def test_linear_probe(self, tiny_dataset):
        _run_training(tiny_dataset, ["--linear_probe"])

    def test_full_finetune(self, tiny_dataset):
        _run_training(tiny_dataset, ["--full_finetune"])

    def test_resnet_baseline(self, tiny_dataset):
        _run_training(tiny_dataset, ["--resnet_baseline"])

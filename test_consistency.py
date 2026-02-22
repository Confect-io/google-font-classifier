#!/usr/bin/env python3
"""
Consistency checks for the font-model repo.

Run before any paper submission or model upload to catch drift between
code, paper, and model config. These are the exact classes of bugs
that have bitten us before.

Usage:
    python test_consistency.py
    python test_consistency.py --model dchen0/font_classifier_v4
    python test_consistency.py --model dchen0/font_classifier_v4 --data_dir .data_out/font_dataset_v4
"""

import argparse
import os
import re
import sys

FONT_ALLOWLIST_EXPECTED_COUNT = 32
EXPECTED_VARIANTS = 394


def test_preprocessing_consistency():
    """Verify train and inference use the same transform function."""
    from handler import get_inference_transform
    from train_model import get_transform
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
    size = processor.size["shortest_edge"]

    # get_transform should use get_inference_transform internally
    import inspect
    source = inspect.getsource(get_transform)
    assert "get_inference_transform" in source, (
        "train_model.get_transform does not use handler.get_inference_transform — "
        "this means training and inference preprocessing may differ (train/serve skew)"
    )

    # Verify the transform produces the same output
    from PIL import Image
    import torch
    img = Image.new("RGB", (400, 100), (128, 128, 128))

    inference_transform = get_inference_transform(processor, size)
    train_transform_fn = get_transform(processor, size)

    inference_result = inference_transform(img)

    # Simulate what train_transform does
    example = {"image": img}
    train_result = train_transform_fn(example)["pixel_values"]

    assert torch.equal(inference_result, train_result), (
        "Inference and training transforms produce different outputs for the same image"
    )
    print("  PASS: preprocessing is consistent between training and inference")


def test_font_allowlist_count():
    """Verify FONT_ALLOWLIST has the expected number of families."""
    from dataset_generator import FONT_ALLOWLIST

    assert len(FONT_ALLOWLIST) == FONT_ALLOWLIST_EXPECTED_COUNT, (
        f"FONT_ALLOWLIST has {len(FONT_ALLOWLIST)} families, expected {FONT_ALLOWLIST_EXPECTED_COUNT}. "
        f"Update the paper if this changed."
    )
    assert len(FONT_ALLOWLIST) == len(set(FONT_ALLOWLIST)), (
        "FONT_ALLOWLIST has duplicate entries"
    )
    print(f"  PASS: FONT_ALLOWLIST has {FONT_ALLOWLIST_EXPECTED_COUNT} unique families")


def test_model_labels_match_allowlist(model_name):
    """Verify model's id2label covers exactly the fonts in FONT_ALLOWLIST."""
    from transformers import AutoConfig
    from dataset_generator import FONT_ALLOWLIST

    config = AutoConfig.from_pretrained(model_name)
    model_labels = set(config.id2label.values())

    # Every model label's family should be in the allowlist
    # Check every model label belongs to some allowlist family
    unmatched_labels = []
    for label in model_labels:
        if not any(label.startswith(fam) for fam in FONT_ALLOWLIST):
            unmatched_labels.append(label)

    assert not unmatched_labels, (
        f"Model has labels not matching any FONT_ALLOWLIST family: {unmatched_labels[:5]}"
    )

    # Check every allowlist family has at least one model label
    unused_families = [
        fam for fam in FONT_ALLOWLIST
        if not any(label.startswith(fam) for label in model_labels)
    ]
    assert not unused_families, (
        f"FONT_ALLOWLIST has families with no model labels: {unused_families}"
    )
    assert len(model_labels) == EXPECTED_VARIANTS, (
        f"Model has {len(model_labels)} labels, expected {EXPECTED_VARIANTS}"
    )
    print(f"  PASS: model {model_name} has {len(model_labels)} labels matching {len(FONT_ALLOWLIST)} allowlist families")


def test_dataset_matches_model(model_name, data_dir):
    """Verify dataset labels match model labels exactly."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    model_labels = set(config.id2label.values())

    test_dir = os.path.join(data_dir, "test")
    assert os.path.isdir(test_dir), f"Test directory not found: {test_dir}"

    dataset_labels = set(
        d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))
    )

    only_model = model_labels - dataset_labels
    only_dataset = dataset_labels - model_labels

    assert not only_model and not only_dataset, (
        f"Model/dataset label mismatch:\n"
        f"  In model but not dataset ({len(only_model)}): {sorted(only_model)[:5]}\n"
        f"  In dataset but not model ({len(only_dataset)}): {sorted(only_dataset)[:5]}"
    )
    print(f"  PASS: dataset has {len(dataset_labels)} classes matching model exactly")


def test_paper_claims():
    """Check that key numbers in the paper match the code."""
    from dataset_generator import FONT_ALLOWLIST

    paper_path = os.path.join(os.path.dirname(__file__), "paper_arxiv.tex")
    if not os.path.exists(paper_path):
        print("  SKIP: paper_arxiv.tex not found")
        return

    with open(paper_path) as f:
        paper = f.read()

    # Check family count
    family_count = len(FONT_ALLOWLIST)
    assert f"{family_count} base font families" in paper or f"{family_count} families" in paper, (
        f"Paper doesn't mention {family_count} families — may be stale"
    )

    # Check variant count
    assert f"{EXPECTED_VARIANTS} font" in paper or f"{EXPECTED_VARIANTS} distinct" in paper, (
        f"Paper doesn't mention {EXPECTED_VARIANTS} variants — may be stale"
    )

    # Check no old wrong numbers
    assert "31 base font families" not in paper, "Paper still says 31 families (should be 32)"
    assert "150K trainable" not in paper, "Paper still says 150K trainable params (should be 900K)"
    assert "less than 0.2\\%" not in paper, "Paper still says less than 0.2% (should be ~1%)"

    print("  PASS: paper claims match code")


def test_no_hardcoded_defaults():
    """Verify confusion_matrix.py doesn't have hardcoded model/dataset defaults."""
    cm_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.py")
    with open(cm_path) as f:
        source = f.read()

    # Check that --model and --data_dir are required
    assert "required=True" in source, (
        "confusion_matrix.py should have required=True for --model and --data_dir"
    )
    assert 'default="dchen0/' not in source, (
        "confusion_matrix.py has a hardcoded default model name"
    )
    assert 'default=".data_out/' not in source, (
        "confusion_matrix.py has a hardcoded default data directory"
    )
    print("  PASS: no hardcoded defaults in confusion_matrix.py")


def main():
    parser = argparse.ArgumentParser(description="Run consistency checks")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name to check against")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset directory to check against model")
    args = parser.parse_args()

    passed = 0
    failed = 0
    skipped = 0

    tests = [
        ("Preprocessing consistency", test_preprocessing_consistency, []),
        ("Font allowlist count", test_font_allowlist_count, []),
        ("Paper claims", test_paper_claims, []),
        ("No hardcoded defaults", test_no_hardcoded_defaults, []),
    ]

    if args.model:
        tests.append(("Model labels match allowlist", test_model_labels_match_allowlist, [args.model]))
    if args.model and args.data_dir:
        tests.append(("Dataset matches model", test_dataset_matches_model, [args.model, args.data_dir]))

    print(f"Running {len(tests)} consistency checks...\n")

    for name, fn, fn_args in tests:
        try:
            fn(*fn_args)
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"  {passed} passed, {failed} failed")
    if failed:
        print(f"  FIX THESE BEFORE SUBMITTING")
        sys.exit(1)
    else:
        print(f"  All checks passed!")


if __name__ == "__main__":
    main()

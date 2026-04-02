"""Lightweight SWER computation for all training modes.

Handles the ._* prefix mapping issue: models trained with 788 classes (394 real +
394 macOS resource fork ._* dirs) may predict ._X instead of X. This script strips
the ._ prefix before comparing predictions to ground truth.

Skips figure generation and t-SNE to minimize runtime.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from collections import defaultdict

# Import shared utilities
from confusion_matrix import (
    compute_typographic_distance_matrix,
    compute_severity_metrics,
    sort_labels_by_family,
    extract_family,
)
from handler import get_inference_transform

DATA_DIR = ".data_out/font_dataset_v4"
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
BATCH_SIZE = 32


def strip_dot_prefix(label):
    """Strip ._ prefix from macOS resource fork labels."""
    if label.startswith("._"):
        return label[2:]
    return label


def load_dinov2_model(model_path, data_dir):
    """Load a DINOv2 model (LoRA adapter or full model)."""
    from confusion_matrix import load_model
    return load_model(model_path, data_dir=data_dir)


def load_resnet_model(checkpoint_path, label_names):
    """Load a ResNet-50 model from a training checkpoint."""
    from torchvision.models import resnet50
    from transformers import PretrainedConfig, AutoImageProcessor
    from safetensors.torch import load_model as load_safetensors_model

    class ResNetForImageClassification(torch.nn.Module):
        def __init__(self, backbone, id2label):
            super().__init__()
            self.backbone = backbone
            self.config = PretrainedConfig()
            self.config.id2label = id2label
            self.config.label2id = {v: k for k, v in id2label.items()}
            self.config.num_labels = len(id2label)

        def forward(self, pixel_values, labels=None):
            logits = self.backbone(pixel_values)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}

    id2label = {i: name for i, name in enumerate(label_names)}
    backbone = resnet50()
    backbone.fc = torch.nn.Linear(backbone.fc.in_features, len(label_names))
    model = ResNetForImageClassification(backbone, id2label)
    load_safetensors_model(model, os.path.join(checkpoint_path, "model.safetensors"))

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
    size = processor.size["shortest_edge"]
    transform = get_inference_transform(processor, size)
    return model, transform


def run_inference(model, transform, test_dir, device, id2label=None):
    """Run inference on all test images, return (y_true, y_pred)."""
    if id2label is None:
        id2label = model.config.id2label

    test_labels = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and not d.startswith(".")
    ])

    y_true, y_pred = [], []
    batch_paths, batch_labels = [], []
    total = 0

    def flush():
        nonlocal batch_paths, batch_labels
        if not batch_paths:
            return
        tensors = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            if hasattr(model, "backbone"):
                # ResNet wrapper
                out = model(pixel_values=batch)
                logits = out["logits"]
            else:
                # DINOv2
                out = model(pixel_values=batch)
                logits = out.logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        pred_labels = [id2label[i] for i in preds]

        for true, pred in zip(batch_labels, pred_labels):
            y_true.append(true)
            # Strip ._ prefix if present
            y_pred.append(strip_dot_prefix(pred))

        batch_paths, batch_labels = [], []

    for label in test_labels:
        class_dir = os.path.join(test_dir, label)
        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            if not os.path.isfile(fpath):
                continue
            batch_paths.append(fpath)
            batch_labels.append(label)
            total += 1
            if len(batch_paths) >= BATCH_SIZE:
                flush()
                print(f"\r  Evaluated {len(y_true)}/{total}+ images ...", end="", flush=True)

    flush()
    print(f"\r  Evaluated {len(y_true)} images total.              ")
    return y_true, y_pred


def compute_swer(y_true, y_pred):
    """Compute SWER using typographic distance matrix."""
    all_labels = sort_labels_by_family(list(set(y_true) | set(y_pred)))
    typo_dist_matrix = compute_typographic_distance_matrix(all_labels)
    severity = compute_severity_metrics(y_true, y_pred, all_labels, typo_dist_matrix)

    raw_acc = 1.0 - severity["n_misclassifications"] / severity["n_total"]
    swa = 1.0 - severity["severity_weighted_error"]
    relative = 0.0
    if severity["random_baseline_severity"] > 0:
        relative = severity["severity_weighted_error"] / severity["random_baseline_severity"]

    return {
        "n_total": severity["n_total"],
        "n_misclassifications": severity["n_misclassifications"],
        "mean_misclass_severity": severity["mean_misclass_severity"],
        "swer": severity["severity_weighted_error"],
        "random_baseline": severity["random_baseline_severity"],
        "relative_severity": relative,
        "top1_accuracy": raw_acc,
        "family_accuracy": severity["family_accuracy"],
        "severity_weighted_accuracy": swa,
    }


def print_results(mode, results):
    """Print SWER results for a mode."""
    print(f"\n{'='*60}")
    print(f"  {mode}")
    print(f"{'='*60}")
    print(f"  Misclassifications:             {results['n_misclassifications']}/{results['n_total']}")
    print(f"  Mean misclassification severity: {results['mean_misclass_severity']:.4f}")
    print(f"  Severity-weighted error rate:    {results['swer']:.4f}")
    print(f"  Random baseline error rate:      {results['random_baseline']:.4f}")
    print(f"  Relative severity (vs random):   {results['relative_severity']:.4f}")
    print(f"  Top-1 accuracy:                  {results['top1_accuracy']:.1%}")
    print(f"  Family-level accuracy:           {results['family_accuracy']:.1%}")
    print(f"  Severity-weighted accuracy:      {results['severity_weighted_accuracy']:.1%}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True,
                   choices=["lora_r4", "lora_r8", "lora_r16", "full_finetune", "linear_probe", "resnet50"])
    p.add_argument("--model", type=str, required=True, help="Path to model/checkpoint")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")

    if args.mode == "resnet50":
        label_names = sorted([
            d for d in os.listdir(TRAIN_DIR)
            if not d.startswith(".") and os.path.isdir(os.path.join(TRAIN_DIR, d))
        ])
        model, transform = load_resnet_model(args.model, label_names)
    else:
        model, transform = load_dinov2_model(args.model, data_dir=DATA_DIR)

    model.to(device)
    model.eval()

    print("Running inference ...")
    y_true, y_pred = run_inference(model, transform, TEST_DIR, device)

    print("Computing SWER ...")
    results = compute_swer(y_true, y_pred)
    print_results(args.mode, results)

    # Save results as JSON
    output_path = f"swer_{args.mode}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

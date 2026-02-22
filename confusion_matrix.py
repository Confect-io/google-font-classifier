"""
Generate confusion matrix figures and per-class metrics for the font classifier.

Produces:
  figures/confusion_matrix.pdf   — full NxN heatmap (normalized)
  figures/top_confused_pairs.pdf — bar chart of most frequent misclassifications
  confusion_matrix.json          — raw counts as nested dict
  bad_images.json                — list of misclassified images
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from transformers import AutoImageProcessor, Dinov2ForImageClassification

from handler import get_inference_transform

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate font classifier and produce figures.")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root of dataset (must contain a test/ subdirectory)")
    p.add_argument("--model", type=str, required=True,
                   help="HuggingFace model name or local path")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for inference")
    p.add_argument("--top_n", type=int, default=20,
                   help="Number of most-confused pairs to plot")
    p.add_argument("--output_dir", type=str, default="figures",
                   help="Directory for output figures")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name):
    model = Dinov2ForImageClassification.from_pretrained(
        model_name, ignore_mismatched_sizes=True,
    )
    processor = AutoImageProcessor.from_pretrained(model_name)
    model.eval()
    size = processor.size["shortest_edge"]
    transform = get_inference_transform(processor, size)
    return model, transform

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_batch(model, transform, image_paths, device):
    tensors = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=batch, output_hidden_states=True)
    preds = outputs.logits.argmax(dim=-1).cpu().tolist()
    # [CLS] token embedding from last hidden layer
    cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
    id2label = model.config.id2label
    return [id2label[i] for i in preds], cls_embeddings

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, transform, test_dir, batch_size, device):
    y_true, y_pred = [], []
    all_embeddings = []
    bad_images = []

    labels = sorted(os.listdir(test_dir))
    labels = [l for l in labels if os.path.isdir(os.path.join(test_dir, l))]

    # Check for empty class directories
    empty_classes = [
        l for l in labels
        if len(os.listdir(os.path.join(test_dir, l))) == 0
    ]
    if empty_classes:
        print(f"  WARNING: {len(empty_classes)} empty test class dirs (will be skipped): {empty_classes[:5]}{'...' if len(empty_classes) > 5 else ''}")
        labels = [l for l in labels if l not in set(empty_classes)]

    batch_paths, batch_labels = [], []

    def flush():
        nonlocal batch_paths, batch_labels
        if not batch_paths:
            return
        preds, cls_embs = predict_batch(model, transform, batch_paths, device)
        all_embeddings.append(cls_embs)
        for path, true, pred in zip(batch_paths, batch_labels, preds):
            y_true.append(true)
            y_pred.append(pred)
            if true != pred:
                bad_images.append({"image": path, "label": true, "output_label": pred})
        batch_paths, batch_labels = [], []

    total_images = 0
    for label in labels:
        class_dir = os.path.join(test_dir, label)
        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            if not os.path.isfile(fpath):
                continue
            batch_paths.append(fpath)
            batch_labels.append(label)
            total_images += 1
            if len(batch_paths) >= batch_size:
                flush()
                print(f"\r  Evaluated {len(y_true)}/{total_images}+ images ...", end="", flush=True)

    flush()
    print(f"\r  Evaluated {len(y_true)} images total.              ")
    embeddings = np.concatenate(all_embeddings, axis=0)
    return y_true, y_pred, bad_images, labels, embeddings

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def extract_family(label):
    """Extract font family from a label like 'Barlow-BoldItalic' or 'Aleo_Bold'."""
    for sep in ["-", "_"]:
        if sep in label:
            return label.split(sep)[0]
    return label


def sort_labels_by_family(labels):
    """Sort labels so that variants of the same family are adjacent."""
    return sorted(labels, key=lambda l: (extract_family(l), l))


def plot_confusion_matrix(y_true, y_pred, sorted_labels, output_path):
    cm = sk_confusion_matrix(y_true, y_pred, labels=sorted_labels)
    # Normalize by row (true label)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    # Draw family group boundaries
    families = [extract_family(l) for l in sorted_labels]
    boundaries = []
    for i in range(1, len(families)):
        if families[i] != families[i - 1]:
            boundaries.append(i)
    for b in boundaries:
        ax.axhline(b - 0.5, color="gray", linewidth=0.3, alpha=0.5)
        ax.axvline(b - 0.5, color="gray", linewidth=0.3, alpha=0.5)

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (row-normalized)", fontsize=14)

    # With 300 classes, individual tick labels are unreadable — add family labels
    # at midpoints instead
    family_mids = {}
    for i, fam in enumerate(families):
        family_mids.setdefault(fam, [])
        family_mids[fam].append(i)
    mid_positions = []
    mid_labels = []
    for fam, idxs in family_mids.items():
        mid_positions.append(np.mean(idxs))
        mid_labels.append(fam)

    ax.set_xticks(mid_positions)
    ax.set_xticklabels(mid_labels, rotation=90, fontsize=5)
    ax.set_yticks(mid_positions)
    ax.set_yticklabels(mid_labels, fontsize=5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_top_confused_pairs(y_true, y_pred, sorted_labels, top_n, output_path):
    cm = sk_confusion_matrix(y_true, y_pred, labels=sorted_labels)
    np.fill_diagonal(cm, 0)  # ignore correct predictions

    # Collect off-diagonal entries
    pairs = []
    for i in range(len(sorted_labels)):
        for j in range(len(sorted_labels)):
            if cm[i, j] > 0:
                pairs.append((sorted_labels[i], sorted_labels[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:top_n]

    if not pairs:
        print("  No misclassifications found — skipping top-confused-pairs chart.")
        return

    labels_str = [f"{a} → {b}" for a, b, _ in pairs]
    counts = [c for _, _, c in pairs]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(pairs))))
    y_pos = np.arange(len(labels_str))
    ax.barh(y_pos, counts, color="#4C72B0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_str, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Top-{top_n} Most Frequent Misclassifications")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")

def plot_tsne_embeddings(y_true, embeddings, output_path):
    from sklearn.manifold import TSNE

    families = [extract_family(l) for l in y_true]
    unique_families = sorted(set(families))

    print("  Running t-SNE (this may take a minute) ...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(embeddings)

    # Combine tab20 + tab20b to get 40 distinct colors (we have 32 families)
    tab20 = plt.cm.get_cmap("tab20")
    tab20b = plt.cm.get_cmap("tab20b")
    cmap = lambda i: tab20(i) if i < 20 else tab20b(i - 20)
    fig, ax = plt.subplots(figsize=(12, 10))

    for i, fam in enumerate(unique_families):
        mask = [f == fam for f in families]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=8, alpha=0.6, color=cmap(i), label=fam,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of [CLS] Embeddings (colored by font family)")
    ax.legend(
        fontsize=5, ncol=3, markerscale=2,
        bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_sample_images(test_dir, output_path):
    # Pick ~4 families spanning different categories
    target_families = ["CrimsonPro", "Inter", "JetBrainsMono", "BigShouldersText"]

    # Discover available families and their variants in test_dir
    all_labels = sorted(
        d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))
    )
    family_variants = defaultdict(list)
    for label in all_labels:
        family_variants[extract_family(label)].append(label)

    # Filter to families that actually exist in the dataset
    families_to_show = [f for f in target_families if f in family_variants]
    # Fall back: if fewer than 4 matched, fill from available families
    if len(families_to_show) < 4:
        for fam in sorted(family_variants):
            if fam not in families_to_show:
                families_to_show.append(fam)
            if len(families_to_show) >= 4:
                break

    # For each family, pick up to 3 variants (first, middle, last by sorted order)
    rows = []
    for fam in families_to_show:
        variants = sorted(family_variants[fam])
        if len(variants) <= 3:
            chosen = variants
        else:
            chosen = [variants[0], variants[len(variants) // 2], variants[-1]]
        rows.append((fam, chosen))

    n_rows = len(rows)
    n_cols = max(len(v) for _, v in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows),
                             squeeze=False)

    for r, (fam, variants) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            if c < len(variants):
                variant = variants[c]
                variant_dir = os.path.join(test_dir, variant)
                # Pick the first image file
                imgs = sorted(
                    f for f in os.listdir(variant_dir)
                    if os.path.isfile(os.path.join(variant_dir, f))
                    and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
                )
                if imgs:
                    img = Image.open(os.path.join(variant_dir, imgs[0])).convert("RGB")
                    ax.imshow(img)
                    ax.set_title(variant, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Sample Test Images", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_per_family_accuracy(y_true, y_pred, output_path):
    family_correct = defaultdict(int)
    family_total = defaultdict(int)

    for true, pred in zip(y_true, y_pred):
        fam = extract_family(true)
        family_total[fam] += 1
        if true == pred:
            family_correct[fam] += 1

    families = sorted(family_total.keys())
    accuracies = [family_correct[f] / family_total[f] for f in families]

    # Sort by accuracy
    order = np.argsort(accuracies)
    families = [families[i] for i in order]
    accuracies = [accuracies[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(families))))
    y_pos = np.arange(len(families))
    ax.barh(y_pos, accuracies, color="#4C72B0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(families, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-Family Classification Accuracy")
    # Add value labels
    for i, acc in enumerate(accuracies):
        ax.text(acc + 0.01, i, f"{acc:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_accuracy_vs_variants(y_true, y_pred, sorted_labels, output_path):
    from adjustText import adjust_text

    # Count variants per family
    family_variants = defaultdict(int)
    for label in sorted_labels:
        family_variants[extract_family(label)] += 1

    # Compute accuracy per family
    family_correct = defaultdict(int)
    family_total = defaultdict(int)
    for true, pred in zip(y_true, y_pred):
        fam = extract_family(true)
        family_total[fam] += 1
        if true == pred:
            family_correct[fam] += 1

    families = sorted(family_total.keys())
    n_variants = [family_variants[f] for f in families]
    accuracies = [family_correct[f] / family_total[f] for f in families]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(n_variants, accuracies, s=50, color="#4C72B0", edgecolors="black", linewidths=0.5)

    texts = []
    for fam, nv, acc in zip(families, n_variants, accuracies):
        texts.append(ax.text(nv, acc, fam, fontsize=6))

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax.set_xlabel("Number of Variants")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Family Accuracy vs. Number of Variants")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Similarity & severity analysis
# ---------------------------------------------------------------------------

def compute_class_centroids_and_distances(y_true, embeddings, all_labels):
    """Compute mean [CLS] embedding per class and pairwise cosine distance matrix."""
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n_classes = len(all_labels)
    dim = embeddings.shape[1]

    # Accumulate embeddings per class
    sums = np.zeros((n_classes, dim))
    counts = np.zeros(n_classes)
    for label, emb in zip(y_true, embeddings):
        idx = label_to_idx[label]
        sums[idx] += emb
        counts[idx] += 1

    # Compute centroids (add epsilon to avoid zero-norm for classes with no samples)
    centroids = np.zeros((n_classes, dim))
    for i in range(n_classes):
        if counts[i] > 0:
            centroids[i] = sums[i] / counts[i]
        else:
            centroids[i] = 1e-10  # epsilon

    # Pairwise cosine distances, normalized to [0, 1]
    dist_condensed = pdist(centroids, metric="cosine")
    dist_matrix = squareform(dist_condensed)
    max_dist = dist_matrix.max()
    if max_dist > 0:
        dist_matrix /= max_dist
    return centroids, dist_matrix


def compute_family_centroids_and_distances(centroids, all_labels):
    """Aggregate class centroids to family level and compute family distance matrix."""
    family_map = defaultdict(list)
    for i, label in enumerate(all_labels):
        family_map[extract_family(label)].append(i)

    family_names = sorted(family_map.keys())
    family_centroids = np.zeros((len(family_names), centroids.shape[1]))
    for i, fam in enumerate(family_names):
        family_centroids[i] = centroids[family_map[fam]].mean(axis=0)

    dist_condensed = pdist(family_centroids, metric="cosine")
    family_dist_matrix = squareform(dist_condensed)
    max_dist = family_dist_matrix.max()
    if max_dist > 0:
        family_dist_matrix /= max_dist
    return family_centroids, family_dist_matrix, family_names


def compute_severity_metrics(y_true, y_pred, all_labels, dist_matrix):
    """Compute severity-weighted error metrics using embedding distances."""
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    severities = []
    misclass_severities = []
    family_correct = 0

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            severities.append(0.0)
            family_correct += 1
        else:
            d = dist_matrix[label_to_idx[true], label_to_idx[pred]]
            severities.append(d)
            misclass_severities.append(d)
            if extract_family(true) == extract_family(pred):
                family_correct += 1

    n = len(all_labels)
    # Expected severity under uniform random predictions (mean of full distance matrix)
    random_baseline = dist_matrix.sum() / (n * n)

    return {
        "n_total": len(y_true),
        "n_misclassifications": len(misclass_severities),
        "mean_misclass_severity": np.mean(misclass_severities) if misclass_severities else 0.0,
        "severity_weighted_error": np.mean(severities),
        "random_baseline_severity": random_baseline,
        "family_accuracy": family_correct / len(y_true),
    }


def plot_dendrogram(dist_matrix, labels, output_path, title="Font Similarity Dendrogram"):
    """Plot a hierarchical clustering dendrogram from a distance matrix."""
    dist_condensed = squareform(dist_matrix)
    Z = linkage(dist_condensed, method="average")

    fig_height = max(6, 0.3 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    scipy_dendrogram(
        Z, labels=labels, orientation="left", leaf_font_size=7, ax=ax,
        color_threshold=0,  # single color for clean look
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Cosine Distance (normalized)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def print_metrics(y_true, y_pred, all_labels):
    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall accuracy: {acc:.4f} ({sum(t == p for t, p in zip(y_true, y_pred))}/{len(y_true)})")
    print("\nPer-class precision / recall / F1 (macro avg at bottom):\n")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0, output_dict=False))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model ...")
    model, transform = load_model(args.model)
    model.to(device)

    # --- Sanity check: verify test labels match model labels ---
    model_labels = set(model.config.id2label.values())
    dir_labels_set = set(
        d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))
    )
    only_in_test = dir_labels_set - model_labels
    only_in_model = model_labels - dir_labels_set
    overlap = dir_labels_set & model_labels

    print(f"\nLabel check: {len(dir_labels_set)} test classes, {len(model_labels)} model classes, {len(overlap)} overlap")
    if only_in_test:
        print(f"  WARNING: {len(only_in_test)} test classes NOT in model: {sorted(only_in_test)[:5]}{'...' if len(only_in_test) > 5 else ''}")
    if only_in_model:
        print(f"  WARNING: {len(only_in_model)} model classes NOT in test: {sorted(only_in_model)[:5]}{'...' if len(only_in_model) > 5 else ''}")

    if len(overlap) < len(dir_labels_set) * 0.5:
        raise ValueError(
            f"Model/dataset mismatch: only {len(overlap)}/{len(dir_labels_set)} test classes "
            f"exist in the model's label space. This will produce meaningless accuracy numbers. "
            f"Use a model that was trained on this dataset."
        )
    print()

    print("Evaluating ...")
    y_true, y_pred, bad_images, dir_labels, embeddings = evaluate(
        model, transform, test_dir, args.batch_size, device,
    )

    # Use the union of directory labels and any predicted labels not in dir_labels
    all_labels = sort_labels_by_family(list(set(dir_labels) | set(y_pred)))

    # --- Figures ---
    print("Generating figures ...")
    plot_confusion_matrix(
        y_true, y_pred, all_labels,
        os.path.join(args.output_dir, "confusion_matrix.pdf"),
    )
    plot_top_confused_pairs(
        y_true, y_pred, all_labels, args.top_n,
        os.path.join(args.output_dir, "top_confused_pairs.pdf"),
    )
    plot_tsne_embeddings(
        y_true, embeddings,
        os.path.join(args.output_dir, "tsne_embeddings.pdf"),
    )
    plot_sample_images(
        test_dir,
        os.path.join(args.output_dir, "sample_images.pdf"),
    )
    plot_per_family_accuracy(
        y_true, y_pred,
        os.path.join(args.output_dir, "per_family_accuracy.pdf"),
    )
    plot_accuracy_vs_variants(
        y_true, y_pred, all_labels,
        os.path.join(args.output_dir, "accuracy_vs_variants.pdf"),
    )

    # --- Similarity & severity analysis ---
    print("Computing embedding-based similarity ...")
    centroids, dist_matrix = compute_class_centroids_and_distances(
        y_true, embeddings, all_labels,
    )
    family_centroids, family_dist_matrix, family_names = compute_family_centroids_and_distances(
        centroids, all_labels,
    )
    plot_dendrogram(
        family_dist_matrix, family_names,
        os.path.join(args.output_dir, "font_dendrogram.pdf"),
        title="Font Family Similarity (UPGMA of [CLS] Embeddings)",
    )
    plot_dendrogram(
        dist_matrix, all_labels,
        os.path.join(args.output_dir, "font_dendrogram_full.pdf"),
        title="Font Variant Similarity (UPGMA of [CLS] Embeddings)",
    )
    severity = compute_severity_metrics(y_true, y_pred, all_labels, dist_matrix)

    # --- JSON outputs ---
    cm_dict = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        cm_dict[t][p] += 1
    cm_dict = {k: dict(v) for k, v in cm_dict.items()}
    with open("confusion_matrix.json", "w") as f:
        json.dump(cm_dict, f, indent=2)
    print("  Saved confusion_matrix.json")

    with open("bad_images.json", "w") as f:
        json.dump(bad_images, f, indent=2)
    print(f"  Saved bad_images.json ({len(bad_images)} misclassified images)")

    # --- Metrics ---
    print_metrics(y_true, y_pred, all_labels)

    raw_acc = 1.0 - severity['n_misclassifications'] / severity['n_total']
    swa = 1.0 - severity['severity_weighted_error']

    print("\n--- Severity-Weighted Error Analysis ---")
    print(f"  Misclassifications:             {severity['n_misclassifications']}/{severity['n_total']}")
    print(f"  Mean misclassification severity: {severity['mean_misclass_severity']:.4f}  (0=same class, 1=maximally distant)")
    print(f"  Severity-weighted error rate:    {severity['severity_weighted_error']:.4f}")
    print(f"  Random baseline error rate:      {severity['random_baseline_severity']:.4f}")
    relative = 0.0
    if severity['random_baseline_severity'] > 0:
        relative = severity['severity_weighted_error'] / severity['random_baseline_severity']
        print(f"  Relative severity (vs random):   {relative:.4f}")
    print(f"  Top-1 accuracy:                  {raw_acc:.1%}")
    print(f"  Family-level accuracy:           {severity['family_accuracy']:.1%}")
    print(f"  Severity-weighted accuracy:      {swa:.1%}")

    # Write LaTeX macros so paper.tex picks up the values automatically
    metrics_path = os.path.join(args.output_dir, "metrics.tex")
    with open(metrics_path, "w") as f:
        f.write(f"\\newcommand{{\\nMisclass}}{{{severity['n_misclassifications']}}}\n")
        f.write(f"\\newcommand{{\\nTotal}}{{{severity['n_total']}}}\n")
        f.write(f"\\newcommand{{\\meanMisclassSeverity}}{{{severity['mean_misclass_severity']:.4f}}}\n")
        f.write(f"\\newcommand{{\\swerValue}}{{{severity['severity_weighted_error']:.4f}}}\n")
        f.write(f"\\newcommand{{\\swerRandom}}{{{severity['random_baseline_severity']:.4f}}}\n")
        f.write(f"\\newcommand{{\\swerRelative}}{{{relative:.4f}}}\n")
        f.write(f"\\newcommand{{\\rawAccuracy}}{{{raw_acc:.1%}}}".replace("%", "\\%") + "\n")
        f.write(f"\\newcommand{{\\familyAccuracy}}{{{severity['family_accuracy']:.1%}}}".replace("%", "\\%") + "\n")
        f.write(f"\\newcommand{{\\severityWeightedAccuracy}}{{{swa:.1%}}}".replace("%", "\\%") + "\n")
    print(f"  Saved {metrics_path}")


if __name__ == "__main__":
    main()

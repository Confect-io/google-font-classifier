"""Export the trained LoRA-adapted DINOv2 classifier to ONNX.

Downloads the LoRA adapter from a results repo, merges it into the base
DINOv2 model, exports to ONNX at the resolution training used (read from the
base processor, currently 256), and
writes the int->label JSON the design-agent expects next to its fonts.py.

Run:
    uv run --with torch --with transformers --with peft --with onnx \\
        --with safetensors --with requests \\
        python3 export_onnx.py \\
            --results_repo confect/google-font-classifier \\
            --model_path lora_r8/result_model \\
            --dataset_name confect/google-font-dataset \\
            --onnx_out ./font-classifier-v5.onnx \\
            --labels_out ./font_labels.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import requests
import torch
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoImageProcessor, Dinov2ForImageClassification

BASE_MODEL = "facebook/dinov2-base-imagenet1k-1-layer"

# Export at whatever size TRAINING used, which is
# `processor.size["shortest_edge"]` (train_model.py reads exactly that).
#
# This was hardcoded to 224 with a comment claiming the processor said 224.
# It says **256**; `crop_size` is the 224 (and train_model.py never crops).
# So every export before this ran the model at a resolution it had never
# seen. DINOv2 interpolates its position embeddings, so it degrades silently
# rather than failing: measured 89.1% top-1 at 224 against 96.3% at 256 on
# the same v6 checkpoint. Read it from the processor instead of asserting it.
def _training_input_size() -> int:
    proc = AutoImageProcessor.from_pretrained(BASE_MODEL)
    size = proc.size["shortest_edge"]
    print(f"Training input size from processor: {size}")
    return size


def get_num_labels_from_checkpoint(adapter_path: str) -> int:
    safetensors_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(safetensors_file):
        safetensors_file = os.path.join(adapter_path, "model.safetensors")
    with safe_open(safetensors_file, framework="pt") as f:
        for key in f.keys():
            if "classifier" in key and "weight" in key:
                return f.get_tensor(key).shape[0]
    raise RuntimeError("Could not infer num_labels from classifier weights")


def get_label_names_from_dataset(dataset_name: str, token: str | None = None) -> list[str]:
    """Stream tar headers to read the imagefolder class-dir names without
    pulling the full ~40 GB dataset."""
    api = HfApi(token=token)
    files = api.list_repo_files(dataset_name, repo_type="dataset")
    tar_files = [f for f in files if f.endswith(".tar")]

    labels: set[str] = set()
    for tarname in tar_files:
        url = hf_hub_url(dataset_name, tarname, repo_type="dataset")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        headers["Range"] = "bytes=0-52428800"
        r = requests.get(url, headers=headers)
        data = r.content

        pos = 0
        while pos + 512 <= len(data):
            header = data[pos : pos + 512]
            if header == b"\x00" * 512:
                break
            name = header[:100].split(b"\x00")[0].decode("utf-8", errors="ignore")
            typeflag = header[156:157]
            try:
                size = int(header[124:136].strip(b"\x00 "), 8)
            except ValueError:
                break
            if typeflag == b"5":
                m = re.match(r"^(?:test|train)/([^/]+)/?$", name)
                if m:
                    labels.add(m.group(1))
            pos += 512 + ((size + 511) // 512) * 512
        if labels:
            break
    return sorted(labels)


def build_label_list(real_labels: list[str], num_labels: int) -> tuple[list[str], list[str]]:
    """Reconcile dataset label count with classifier weight count (handles
    the macOS ._* resource-fork bug from older runs)."""
    if num_labels == len(real_labels):
        return real_labels, real_labels
    if num_labels == len(real_labels) * 2:
        all_labels = sorted(real_labels + [f"._{l}" for l in real_labels])
        return all_labels, real_labels
    raise ValueError(
        f"Cannot reconcile {num_labels} checkpoint labels with "
        f"{len(real_labels)} dataset labels"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_repo", default="confect/google-font-classifier")
    parser.add_argument("--model_path", default="lora_r8/result_model")
    parser.add_argument(
        "--font_dir",
        type=Path,
        default=Path("./fonts"),
        help="Local fonts/ dir; class labels are the alphabetically-sorted "
             "SUBDIRECTORY names (one per family), matching how "
             "dataset_generator names class folders and how imagefolder "
             "orders them.",
    )
    parser.add_argument("--onnx_out", type=Path, default=Path("./font-classifier-v5.onnx"))
    parser.add_argument("--labels_out", type=Path, default=Path("./font_labels.json"))
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")

    print(f"Downloading {args.model_path} from {args.results_repo} ...")
    local = snapshot_download(
        repo_id=args.results_repo,
        repo_type="model",
        allow_patterns=[f"{args.model_path}/*"],
        token=token,
    )
    adapter_path = os.path.join(local, args.model_path)

    num_labels = get_num_labels_from_checkpoint(adapter_path)
    print(f"Checkpoint classifier has {num_labels} outputs")

    if not args.font_dir.is_dir():
        raise SystemExit(
            f"--font_dir {args.font_dir} not found. Either point it at the "
            f"local fonts/ used for dataset gen, or regenerate via "
            f"curate_fonts.py."
        )
    # One class per FAMILY directory. Weight is an in-class augmentation now,
    # so the .ttf files inside a directory are all the same label.
    real_labels = sorted(p.name for p in args.font_dir.iterdir()
                         if p.is_dir() and any(p.glob("*.ttf")))
    print(f"Found {len(real_labels)} class labels from {args.font_dir}")

    all_labels, deploy_labels = build_label_list(real_labels, num_labels)
    if len(all_labels) != len(deploy_labels):
        print(
            f"Stripping macOS ._* labels: training had {len(all_labels)} "
            f"classifier outputs, deploy keeps {len(deploy_labels)}"
        )

    print("Loading base DINOv2 + LoRA adapter ...")
    base = Dinov2ForImageClassification.from_pretrained(
        BASE_MODEL, num_labels=len(all_labels), ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    print("Merging LoRA weights into base ...")
    merged = model.merge_and_unload()

    if len(all_labels) != len(deploy_labels):
        real_idx = torch.tensor(
            [i for i, name in enumerate(all_labels) if not name.startswith("._")]
        )
        merged.classifier.weight = torch.nn.Parameter(merged.classifier.weight[real_idx])
        merged.classifier.bias = torch.nn.Parameter(merged.classifier.bias[real_idx])

    merged.config.id2label = {i: name for i, name in enumerate(deploy_labels)}
    merged.config.label2id = {name: i for i, name in enumerate(deploy_labels)}
    merged.config.num_labels = len(deploy_labels)
    merged.eval()

    input_size = _training_input_size()
    print(f"Exporting to ONNX at {input_size}x{input_size} input -> {args.onnx_out}")
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        merged,
        dummy,
        str(args.onnx_out),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = args.onnx_out.stat().st_size / (1024 * 1024)
    print(f"ONNX written ({size_mb:.1f} MB)")

    labels_json = {str(i): name for i, name in enumerate(deploy_labels)}
    args.labels_out.write_text(json.dumps(labels_json))
    print(f"Labels written to {args.labels_out} ({len(labels_json)} entries)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

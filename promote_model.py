"""Promote a trained model from the results repo to a deployable HuggingFace repo.

The results repo (e.g. dchen0/font-model-results) stores raw training artifacts:
LoRA adapters, checkpoints, and logs. These aren't directly loadable by HF Inference.

This script merges a LoRA adapter into the base DINOv2 model, attaches label mappings
from the dataset, bundles the inference handler, and uploads a self-contained model
repo that HF Inference can auto-deploy.

Usage:
    python promote_model.py \\
        --results_repo dchen0/font-model-results \\
        --model_path lora_r8/result_model \\
        --dataset_name dchen0/font_crops_v5 \\
        --deploy_repo dchen0/font-classifier

    # For non-LoRA models (full finetune, linear probe):
    python promote_model.py \\
        --results_repo dchen0/font-model-results \\
        --model_path full_finetune/result_model \\
        --dataset_name dchen0/font_crops_v5 \\
        --deploy_repo dchen0/font-classifier \\
        --no_lora
"""

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, snapshot_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoImageProcessor, Dinov2ForImageClassification

MODEL = "facebook/dinov2-base-imagenet1k-1-layer"


def get_num_labels_from_checkpoint(adapter_path):
    """Infer num_labels from the classifier weight shape in the checkpoint."""
    safetensors_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(safetensors_file):
        safetensors_file = os.path.join(adapter_path, "model.safetensors")
    with safe_open(safetensors_file, framework="pt") as f:
        for key in f.keys():
            if "classifier" in key and "weight" in key:
                return f.get_tensor(key).shape[0]
    return None


def get_label_names_from_dataset(dataset_name, token=None):
    """Get sorted label names by streaming tar headers (no full download)."""
    api = HfApi(token=token)
    files = api.list_repo_files(dataset_name, repo_type="dataset")
    tar_files = [f for f in files if f.endswith(".tar")]

    labels = set()
    for tarname in tar_files:
        url = hf_hub_url(dataset_name, tarname, repo_type="dataset")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        headers["Range"] = "bytes=0-52428800"  # 50MB is plenty for directory entries
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
            break  # one tar with labels is enough

    return sorted(labels)


def build_label_list(real_labels, num_labels):
    """Build the full label list, handling the macOS ._* resource fork bug.

    Models trained before the ._* cleanup had both real directories and their
    macOS resource fork counterparts (e.g. both 'Arial' and '._Arial') counted
    as labels. This reconstructs that full sorted list so we can load the
    checkpoint correctly, then strip the ._* entries for deployment.
    """
    if num_labels == len(real_labels):
        return real_labels, real_labels

    if num_labels == len(real_labels) * 2:
        # Reconstruct: for each real label, add a ._<label> counterpart
        all_labels = sorted(real_labels + [f"._{l}" for l in real_labels])
        return all_labels, real_labels

    raise ValueError(
        f"Cannot reconcile {num_labels} checkpoint labels with "
        f"{len(real_labels)} dataset labels"
    )


def main():
    parser = argparse.ArgumentParser(description="Promote a trained model to a deployable HF repo")
    parser.add_argument("--results_repo", required=True,
                        help="HF repo with training results (e.g. dchen0/font-model-results)")
    parser.add_argument("--model_path", required=True,
                        help="Path within results repo to the model (e.g. lora_r8/result_model)")
    parser.add_argument("--dataset_name", required=True,
                        help="HF dataset to get label names from (e.g. dchen0/font_crops_v5)")
    parser.add_argument("--deploy_repo", required=True,
                        help="HF repo to deploy to (e.g. dchen0/font-classifier)")
    parser.add_argument("--no_lora", action="store_true",
                        help="Model is not a LoRA adapter (full finetune or linear probe)")
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_API_KEY")
    api = HfApi(token=token)

    # Download the adapter/model from results repo
    print(f"Downloading {args.model_path} from {args.results_repo}...")
    local_model = snapshot_download(
        repo_id=args.results_repo,
        repo_type="model",
        allow_patterns=[f"{args.model_path}/*"],
        token=token,
    )
    adapter_path = os.path.join(local_model, args.model_path)

    # Infer num_labels from checkpoint
    num_labels = get_num_labels_from_checkpoint(adapter_path)
    print(f"Checkpoint has {num_labels} labels")

    # Get real label names from dataset
    print(f"Loading label names from {args.dataset_name}...")
    real_labels = get_label_names_from_dataset(args.dataset_name, token=token)
    print(f"Found {len(real_labels)} labels in dataset")

    # Build full label list (handling ._* bug if needed)
    all_labels, deploy_labels = build_label_list(real_labels, num_labels)
    if len(all_labels) != len(deploy_labels):
        print(f"Detected macOS ._* resource fork bug: {len(all_labels)} training labels -> {len(deploy_labels)} real labels")

    if args.no_lora:
        print("Loading model (no LoRA merge needed)...")
        merged = Dinov2ForImageClassification.from_pretrained(
            adapter_path,
            ignore_mismatched_sizes=True,
        )
    else:
        print("Loading base model + LoRA adapter...")
        base = Dinov2ForImageClassification.from_pretrained(
            MODEL,
            num_labels=len(all_labels),
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        print("Merging LoRA weights into base model...")
        merged = model.merge_and_unload()

    # Strip ._* label rows from classifier if needed
    if len(all_labels) != len(deploy_labels):
        import torch

        real_indices = [i for i, name in enumerate(all_labels) if not name.startswith("._")]
        idx = torch.tensor(real_indices)
        merged.classifier.weight = torch.nn.Parameter(merged.classifier.weight[idx])
        merged.classifier.bias = torch.nn.Parameter(merged.classifier.bias[idx])
        print(f"Stripped classifier from {len(all_labels)} to {len(deploy_labels)} classes")

    # Set label mappings and pipeline tag
    id2label = {i: name for i, name in enumerate(deploy_labels)}
    label2id = {name: i for i, name in enumerate(deploy_labels)}
    merged.config.id2label = id2label
    merged.config.label2id = label2id
    merged.config.num_labels = len(deploy_labels)
    merged.config.pipeline_tag = "image-classification"

    processor = AutoImageProcessor.from_pretrained(MODEL)

    # Save and upload
    with tempfile.TemporaryDirectory() as tmp:
        print("Saving merged model...")
        merged.save_pretrained(tmp, safe_serialization=True)
        processor.save_pretrained(tmp)

        # Bundle handler and requirements (same as train_model.py)
        script_dir = Path(__file__).parent
        shutil.copy(script_dir / "handler.py", tmp)
        Path(tmp, "requirements.txt").write_text("\n".join([
            "torchvision>=0.19",
            "Pillow>=10",
        ]))

        print(f"Uploading to {args.deploy_repo}...")
        api.create_repo(args.deploy_repo, repo_type="model", exist_ok=True, token=token)
        api.upload_folder(
            repo_id=args.deploy_repo,
            folder_path=tmp,
            commit_message=f"Promote {args.model_path} from {args.results_repo}",
            token=token,
        )

    print(f"Done! Model deployed to: https://huggingface.co/{args.deploy_repo}")


if __name__ == "__main__":
    main()

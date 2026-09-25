import argparse
import json
from pathlib import Path

import torch
from multitask_model import Dinov2ForFontClassification, FontOnnxWrapper
from peft import PeftModel
from transformers import AutoImageProcessor

BASE_MODEL = "facebook/dinov2-base-imagenet1k-1-layer"
INITIAL_ADAPTER = "confect/google-font-classifier-v6"
INITIAL_ADAPTER_SUBFOLDER = "lora_r16/result_model"


def load_model(adapter: Path, metadata: dict):
    base = Dinov2ForFontClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(metadata["family_labels"]),
        ignore_mismatched_sizes=True,
    )
    initial_adapter = metadata.get("initial_adapter", INITIAL_ADAPTER)
    if initial_adapter:
        subfolder = metadata.get(
            "initial_adapter_subfolder", INITIAL_ADAPTER_SUBFOLDER
        )
        adapter_args = {"subfolder": subfolder} if subfolder else {}
        base = PeftModel.from_pretrained(
            base, initial_adapter, **adapter_args
        ).merge_and_unload()
    return PeftModel.from_pretrained(base, adapter).merge_and_unload().eval()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the family-and-weight classifier to ONNX"
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--onnx_out", type=Path, required=True)
    parser.add_argument("--metadata_out", type=Path)
    args = parser.parse_args()

    metadata_path = args.metadata or args.adapter / "font_model_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.setdefault("initial_adapter", INITIAL_ADAPTER)
    metadata.setdefault("initial_adapter_subfolder", INITIAL_ADAPTER_SUBFOLDER)
    labels = metadata["family_labels"]
    model = load_model(args.adapter, metadata)
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    input_size = processor.size["shortest_edge"]

    args.onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        FontOnnxWrapper(model),
        torch.randn(1, 3, input_size, input_size),
        str(args.onnx_out),
        input_names=["pixel_values"],
        output_names=["family_logits", "weight_logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "family_logits": {0: "batch"},
            "weight_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    metadata_path = args.metadata_out or args.onnx_out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":")))
    print(
        f"Exported {args.onnx_out} with {len(labels)} family outputs and "
        f"{len(metadata['weight_thresholds'])} weight outputs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

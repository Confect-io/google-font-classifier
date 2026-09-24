import argparse
import json
from pathlib import Path

import torch
from multitask_model import Dinov2ForFontClassification, FontOnnxWrapper
from peft import PeftModel
from transformers import AutoImageProcessor

BASE_MODEL = "facebook/dinov2-base-imagenet1k-1-layer"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the family-and-weight classifier to ONNX"
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--onnx_out", type=Path, required=True)
    parser.add_argument("--metadata_out", type=Path)
    args = parser.parse_args()

    metadata = json.loads(
        (args.adapter / "font_model_metadata.json").read_text()
    )
    labels = metadata["family_labels"]
    base = Dinov2ForFontClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload().eval()
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

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from handler import get_inference_transform
from export_multitask_onnx import BASE_MODEL, load_model
from PIL import Image
from transformers import AutoImageProcessor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and ONNX family/weight outputs"
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.metadata or args.adapter / "font_model_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    model = load_model(args.adapter, metadata)
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    transform = get_inference_transform(
        processor, processor.size["shortest_edge"]
    )
    with Image.open(args.image) as image:
        pixels = transform(image).unsqueeze(0)

    with torch.no_grad():
        expected = model(pixel_values=pixels)
    session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    family_logits, weight_logits = session.run(
        None, {session.get_inputs()[0].name: pixels.numpy()}
    )
    family_error = float(
        np.max(np.abs(family_logits - expected.logits.numpy()))
    )
    weight_error = float(
        np.max(np.abs(weight_logits - expected.weight_logits.numpy()))
    )
    if family_error >= 1e-4 or weight_error >= 1e-4:
        raise ValueError(
            f"ONNX mismatch: family={family_error}, weight={weight_error}"
        )
    print(
        f"family_logits={family_logits.shape} weight_logits={weight_logits.shape} "
        f"max_abs_error=({family_error:.2g}, {weight_error:.2g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

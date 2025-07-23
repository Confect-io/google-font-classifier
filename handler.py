# to be bundled with the model on upload to HF Inference Endpoints

import base64
import io
from typing import Any, Dict

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoImageProcessor, Dinov2ForImageClassification


def get_inference_transform(processor: AutoImageProcessor, size: int):
    """Get the raw validation transform for direct inference on PIL images."""
    normalize   = T.Normalize(mean=processor.image_mean, std=processor.image_std)

    to_rgb = T.Lambda(lambda img: img.convert('RGB'))
 
    def pad_to_square(img):
        w, h = img.size
        max_size = max(w, h)
        pad_w = (max_size - w) // 2
        pad_h = (max_size - h) // 2
        padding = (pad_w, pad_h, max_size - w - pad_w, max_size - h - pad_h)
        return T.Pad(padding, fill=0)(img)

    aug     = T.Compose([
        to_rgb,
        pad_to_square,
        T.Resize(size),
        T.ToTensor(), 
        normalize
    ])

    return aug


class EndpointHandler:
    """
    HF Inference Endpoints entry‑point.
    Loads model/processor once, then uses your *imported* preprocessing
    on every request.
    """

    def __init__(self, path: str = "", image_size: int = 224):
        # Weights + processor --------------------------------------------------------
        self.processor = AutoImageProcessor.from_pretrained(path or ".")
        self.model     = (
            Dinov2ForImageClassification.from_pretrained(path or ".")
            .eval()
        )

        # Re‑use the exact transform from your code ---------------------------------
        self.transform = get_inference_transform(self.processor, image_size)

        self.id2label = self.model.config.id2label

    # -------------------------------------------------------------------------------
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects {"inputs": "<base64‑encoded image>"}.
        Returns the top prediction + per‑class probabilities.
        """
        # case 1 ─ raw bytes (default HF client / curl -T)
        if isinstance(data, (bytes, bytearray)):
            img_bytes = data

        # case 2 ─ JSON with "inputs": <something>
        elif isinstance(data, dict) and "inputs" in data:
            inp = data["inputs"]

            # Base‑64 string
            if isinstance(inp, str):
                img_bytes = base64.b64decode(inp.split(",")[-1])  # drop "data:..." if present

            # Already‑bytes
            elif isinstance(inp, (bytes, bytearray)):
                img_bytes = inp

            # Already a PIL Image object
            elif hasattr(inp, "convert"):
                image = inp                                    # PIL.Image
            else:
                raise ValueError("Unsupported 'inputs' format")

        else:
            raise ValueError("Unsupported request body type")

        # If we didn’t get a ready‑made PIL Image above, decode bytes → PIL
        if "image" not in locals():
            image = Image.open(io.BytesIO(img_bytes))

        # Preprocess with your own transform
        pixel_values = self.transform(image).unsqueeze(0)   # [1, C, H, W]

        with torch.no_grad():
            logits = self.model(pixel_values).logits[0]        # tensor [num_labels]
            probs  = logits.softmax(dim=-1)

        # convert to the required wire format (top‑k or all classes)
        k = min(5, probs.numel())                              # send top‑5
        topk = torch.topk(probs, k)

        response = [
            {"label": self.id2label[idx.item()], "score": prob.item()}
            for prob, idx in zip(topk.values, topk.indices)
        ]

        return response               # <‑‑ must be a *list* of dicts


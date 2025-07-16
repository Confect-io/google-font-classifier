import json
import os
import sys

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoImageProcessor, Dinov2ForImageClassification

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python serve_model.py <model_path> <image_path>")
        print("  model_path: HuggingFace model name (e.g. 'dchen0/font-classifier-v2')")
        print("  image_path: Path to image file")
        sys.exit(1)
    
    hf_model_name = sys.argv[1]
    image_path = sys.argv[2]
    
        # Regular model loading from HuggingFace
    model = Dinov2ForImageClassification.from_pretrained(
        hf_model_name,
        ignore_mismatched_sizes=True,
    )
    processor = AutoImageProcessor.from_pretrained(hf_model_name)
    
    # Set model to evaluation mode
    model.eval()

    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(pixel_values=inputs['pixel_values'])
        logits = outputs.logits

    # Get prediction probabilities
    label_names = model.config.id2label
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top_5_predictions = [label_names[i] for i in torch.topk(logits, k=5).indices.tolist()[0]]
    top_5_confidences = [probabilities[0][i].item() for i in torch.topk(logits, k=5).indices.tolist()[0]]

    # Print results
    print("\n\n\n--------------------------------")
    print(f"Model: {hf_model_name}")
    print(f"Image: {image_path}\n\n")
    print(f"Top 5 predictions: {top_5_predictions}")
    print(f"Top 5 confidences: {top_5_confidences}")

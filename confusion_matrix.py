import json
import os
import sys

import torch
import torchvision.transforms as T
from peft import PeftModel
from PIL import Image
from transformers import AutoImageProcessor, Dinov2ForImageClassification

# Import the inference transform function from train_model.py
from train_model import get_inference_transform

hf_model_name = "dchen0/font-classifier-v4"

# Regular model loading from HuggingFace
model = Dinov2ForImageClassification.from_pretrained(
    hf_model_name,
    ignore_mismatched_sizes=True,
)
processor = AutoImageProcessor.from_pretrained(hf_model_name)

# Set model to evaluation mode
model.eval()

# Create the transform pipeline that matches training
size = processor.size["shortest_edge"]  # Should be 224
transform = get_inference_transform(processor, size)

def query(image_path):
    # Load and process the image using the same pipeline as training
    image = Image.open(image_path).convert('RGB')
    
    # Apply the same transformations as during training
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

    # Get prediction probabilities
    label_names = model.config.id2label
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top_5_predictions = [label_names[i] for i in torch.topk(logits, k=5).indices.tolist()[0]]
    # top_5_confidences = [probabilities[0][i].item() for i in torch.topk(logits, k=5).indices.tolist()[0]]
    return top_5_predictions[0]

matches = 0
total = 0

from collections import defaultdict

confusion_matrix = defaultdict(lambda: defaultdict(lambda: 0))
bad_images = []

for label in os.listdir("v4/dataset/test"):
	for image in os.listdir(f"v4/dataset/test/{label}"):
		output_label = query(f"v4/dataset/test/{label}/{image}")

		if output_label == label:
			matches += 1
		else:
			bad_images.append({
				"image": f"v4/dataset/test/{label}/{image}",
				"output_label": output_label,
				"label": label
			})
			confusion_matrix[label][output_label] += 1
			confusion_matrix[output_label][label] += 1
		total += 1

		print(f"{matches}/{total}")

# Convert defaultdict to regular dict for JSON serialization
confusion_matrix_dict = {
    label: dict(predictions) for label, predictions in confusion_matrix.items()
}

# Save confusion matrix to JSON file
with open('confusion_matrix.json', 'w') as f:
    json.dump(confusion_matrix_dict, f, indent=2)

with open('bad_images.json', 'w') as f:
    json.dump(bad_images, f, indent=2)

print(f"Confusion matrix saved to confusion_matrix.json")
print(f"Overall accuracy: {matches}/{total} = {matches/total:.2%}")

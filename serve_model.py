import os
import sys

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoImageProcessor, Dinov2ForImageClassification

if __name__ == "__main__":
    # Load the base model and PEFT adapter
    ### TODO how can we embed the label names into the model? 
    label_names = os.listdir("glyphs224_with_subfonts/train")
    ## TODO dedupe the two lines below, similar to train_model.py
    base_model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer",
                                                              num_labels=len(label_names),
                                                              ignore_mismatched_sizes=True,
                                                              )
    model = PeftModel.from_pretrained(base_model, "dchen0/font-classifier", is_trainable=False)
    
    # Merge the PEFT weights into the base model for inference
    model = model.merge_and_unload()
    
    # Load the image processor
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get image path from command line argument or use default
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
    else:
        print("Usage: python serve_model.py <image_path>")
        print("Please provide an image file path as an argument.")
        sys.exit(1)

    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(pixel_values=inputs['pixel_values'])
        logits = outputs.logits


    # Get prediction probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top_5_predictions = [label_names[i] for i in torch.topk(logits, k=5).indices.tolist()[0]]
    top_5_confidences = [probabilities[0][i].item() for i in torch.topk(logits, k=5).indices.tolist()[0]]

    # Print results
    print("\n\n\n--------------------------------")
    print(f"Image: {image_path}\n\n")
    print(f"Top 5 predictions: {top_5_predictions}")
    print(f"Top 5 confidences: {top_5_confidences}")

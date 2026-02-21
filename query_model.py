import io

import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoImageProcessor, Dinov2ForImageClassification

from handler import get_inference_transform

# Initialize FastAPI app
app = FastAPI(title="Font Classifier API", description="API for classifying font types from images")

hf_model_name = "dchen0/font_classifier_v4"

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

def predict_font(image: Image.Image):
    """Helper function to predict font from PIL Image"""
    try:
        # Convert image to RGB if not already
        image = image.convert('RGB')
        
        # Apply the same transformations as during training
        pixel_values = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

        # Get prediction probabilities
        label_names = model.config.id2label
        predictions = torch.softmax(logits, dim=-1)
        top_5_indices = torch.topk(logits, k=5).indices.tolist()[0]
        
        # Format results with confidence scores
        results = []
        for idx in top_5_indices:
            label = label_names[idx]
            score = float(predictions[0][idx])
            results.append({
                "label": label,
                "score": score
            })
        
        return results  # Return top 5
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Font Classifier API is running"}

@app.post("/predict")
async def predict_font_endpoint(file: UploadFile = File(...)):
    """
    Upload an image and get font classification prediction
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Get prediction
        predictions = predict_font(image)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


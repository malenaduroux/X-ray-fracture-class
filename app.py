import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from keras_image_helper import create_preprocessor
import logging
logging.basicConfig(level=logging.INFO)



app = FastAPI(title="fracture-classifier")

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_pytorch_style(X):
    X = X.astype(np.float32) / 255.0
    X = X.transpose(0, 3, 1, 2)

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    return ((X - mean) / std).astype(np.float32)

preprocessor = create_preprocessor(
    preprocess_pytorch_style,
    target_size=(224, 224)
)

# -----------------------------
# Load ONNX model
# -----------------------------
session = ort.InferenceSession(
    "fracture_classifier.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# -----------------------------
# Request / Response models
# -----------------------------
class PredictRequest(BaseModel):
    url: HttpUrl

class PredictResponse(BaseModel):
    fracture_probability: float
    message: str

# -----------------------------
# Prediction
# -----------------------------
def predict(url: str):
    X = preprocessor.from_url(url)
    result = session.run([output_name], {input_name: X})
    logit = result[0][0][0]
    prob = 1 / (1 + np.exp(-logit))  # sigmoid

    fracture_prob = 1 - prob
    return fracture_prob



@app.get("/")
def root():
    return {"message": "Fracture Classification Service"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    fracture_prob = predict(str(request.url))

    message = (
        "This X-ray is likely to depict a fractured arm"
        if fracture_prob > 0.5
        else "This X-ray is unlikely to depict a fractured arm"
    )

    return PredictResponse(
        fracture_probability=fracture_prob,
        message=message
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

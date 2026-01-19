import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from keras_image_helper import create_preprocessor

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
    "fracture_classifier-test.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# -----------------------------
# Request / Response models
# -----------------------------
class PredictRequest(BaseModel):
    url: str = None
    image_path: str = None


class PredictResponse(BaseModel):
    fracture_probability: float
    not_fractured_probability: float


# -----------------------------
# Prediction
# -----------------------------
def predict_from_url(url: str):
    X = preprocessor.from_url(url)
    result = session.run([output_name], {input_name: X})
    logit = result[0][0][0]
    prob = 1 / (1 + np.exp(-logit))  # sigmoid

    # 🔁 Inversion applied here
    fracture_prob = 1 - prob
    return fracture_prob


def predict_from_path(path: str):
    X = preprocessor.from_path(path)
    result = session.run([output_name], {input_name: X})
    logit = result[0][0][0]
    prob = 1 / (1 + np.exp(-logit))

    # 🔁 Inversion applied here
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

    if request.image_path:
        fracture_prob = predict_from_path(request.image_path)
    elif request.url:
        fracture_prob = predict_from_url(request.url)
    else:
        return {"error": "Please provide a url or image_path"}

    return PredictResponse(
        fracture_probability=float(fracture_prob),
        not_fractured_probability=float(1 - fracture_prob)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

import numpy as np
import onnxruntime as ort
from keras_image_helper import create_preprocessor
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, HttpUrl
import uvicorn

app = FastAPI(title="fracture-classifier")

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_pytorch_style(X):
    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    X = X.transpose(0, 3, 1, 2)
    X = (X - mean) / std

    return X.astype(np.float32)

preprocessor = create_preprocessor(
    preprocess_pytorch_style,
    target_size=(224, 224)
)

# ----------------------------
# ONNX Runtime Session
# ----------------------------
session = ort.InferenceSession(
    "fracture_classifier.onnx", providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = [
    "fractured",
    "not fractured"
]

# ----------------------------
# Request + Response Models
# ----------------------------
class PredictRequest(BaseModel):
    url: HttpUrl

class PredictResponse(BaseModel):
    predictions: dict[str, float]
    top_class: str
    top_probability: float

# ----------------------------
# Helper predict function
# ----------------------------
def predict_from_numpy(X):
    result = session.run([output_name], {input_name: X})
    float_predictions = result[0][0].tolist()

    predictions_dict = dict(zip(classes, float_predictions))
    top_class = max(predictions_dict, key=predictions_dict.get)
    top_probability = predictions_dict[top_class]

    return predictions_dict, top_class, top_probability

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/")
def root():
    return {"message": "Fracture Classification Service"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    X = preprocessor.from_url(str(request.url))
    predictions, top_class, top_prob = predict_from_numpy(X)

    return PredictResponse(
        predictions=predictions,
        top_class=top_class,
        top_probability=top_prob
    )

@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    contents = await file.read()

    from PIL import Image
    from io import BytesIO

    img = Image.open(BytesIO(contents)).convert("RGB")
    X = preprocessor.from_pil(img)

    predictions, top_class, top_prob = predict_from_numpy(X)

    return PredictResponse(
        predictions=predictions,
        top_class=top_class,
        top_probability=top_prob
    )

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

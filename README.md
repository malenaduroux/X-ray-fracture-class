# X-ray Fracture Classification

A deep learning project that classifies X-ray images as **fracture** or **not fracture** using EfficientNet and PyTorch.  
Built as a Capstone Project for the **Data Talks Club Machine Learning Zoomcamp 2025**.

---

## 🚨 Problem Statement

Fracture detection in X-ray images is a time-sensitive clinical task.  
A fast, automated screening tool can help healthcare workers triage cases, reduce diagnostic delays, and allocate resources more efficiently.

This project aims to build a binary classifier that predicts whether an X-ray image contains a fracture or not.  
A probability threshold of **0.5** is used to classify images as **fracture** vs **non-fracture**, but this can be adjusted depending on clinical needs.

---

## 📦 Dataset

The dataset is obtained from Kaggle:

**Bone Fracture X-ray Dataset**  
Source: `usman44m/bone-fracture-x-ray-dataset`

It contains X-ray images grouped into two categories:

- `fractured`
- `not fractured`

The data is organized into `train`, `val`, and `test` folders.

---

## 🧠 Approach

A pretrained **EfficientNet-B0** model was fine-tuned for binary classification.  
The final classifier layer was replaced with:

- Dense layer (hidden dim)
- ReLU
- Dropout
- Output layer (single neuron)

### Training Details
- Loss: **BCEWithLogitsLoss**
- Optimizer: **Adam**
- Metrics: **Accuracy, Loss**
- Early stopping based on validation loss
- Checkpointing the best model

The model is saved as `model.pth` and exported to ONNX.

---

## 📌 Output Files

During training, the following are generated:

- `training_curves.png` — loss & accuracy plots  
- `checkpoints/` — best model checkpoints  
- `model.pth` — best saved model  
- `fracture_classifier.onnx` — ONNX exported model  

---

## 🚀 How to Use This Repository

### 1. Clone the repo

```bash
git clone https://github.com/malenaduroux/X-ray-fracture-class.git
cd X-ray-fracture-classification
```


### 🐳 Option A: Run Using Docker
TO BE ADDED
1. Build the Docker image

```bash
docker build -t fracture-prediction .
```

2. Run the container in interactive mode (remove it after using)

```bash
docker run -it --rm -p 8080:8080 fracture-prediction
```

Note: if you want to run the train.py file, you will have to install dev dependencies too: 
```bash
uv sync --locked --extras dev
```

### 💾 Option B: Run locally using uv

1. Install uv if not installed yet
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and sync virtual environment
```bash
uv sync
```

Note: if you want to run the train.py file, you will have to install dev dependencies too: 
```bash
uv sync --locked --extras dev
```

3. Activate it
```bash
source .venv/bin/activate
```

4. Start the FastAPI app
```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```
The app is now available on http://localhost:8080/app like with the Docker method.

You can then upload this text image: https://raw.githubusercontent.com/malenaduroux/X-ray-fracture-class/main/test_image_fractured.jpg to see the service working. Or run the test file:

```bash
uv run python test.py
```

## Cloud Deployment

TO BE ADDED

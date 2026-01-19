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
git clone https://github.com/malenaduroux/X-ray-fracture-classification
cd X-ray-fracture-classification

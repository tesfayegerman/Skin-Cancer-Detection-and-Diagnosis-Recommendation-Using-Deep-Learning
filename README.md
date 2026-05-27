# 🔬 DermAI — Skin Cancer Detection and Diagnosis Recommendation

> A deep learning web application for automated skin cancer classification using EfficientNetB0, trained on the HAM10000 dataset across 7 lesion classes with 85.8% accuracy.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [References](#references)
- [Author](#author)

---

## 🧠 Overview

**DermAI** is a web-based skin cancer detection system that uses transfer learning with EfficientNetB0 to classify dermoscopic images into 7 distinct skin lesion categories. The system accepts image uploads through a browser interface and returns real-time probability scores for each class, helping support early diagnosis decisions.

This project was developed as a final year AI project at **Jimma University (JKU)**, Department of Computer Science, 2026.

---

## 🎯 Features

- ✅ **7-class skin lesion classification** — akiec, bcc, bkl, df, mel, nv, vasc
- ✅ **Real-time prediction** — upload an image and get instant results
- ✅ **Probability chart** — interactive bar chart showing confidence for each class
- ✅ **Balanced training** — oversampled dataset with 1,500 images per class
- ✅ **Focal loss** — handles class imbalance during training
- ✅ **Two model comparison** — EfficientNetB0 vs ResNet50 baseline
- ✅ **Web deployment** — Flask backend with premium HTML/CSS/JS frontend
- ✅ **Lightweight model** — only 4.8M parameters, deployable on CPU

---

## 📊 Dataset

| Property | Details |
|---|---|
| Name | HAM10000 (Human Against Machine with 10,000 Training Images) |
| Source | [Kaggle — HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) |
| Total images | 10,015 original → 10,500 after balancing |
| Image size | 224 × 224 pixels |
| Classes | 7 skin lesion types |
| Split | 70% train / 15% validation / 15% test |

### Class Distribution After Balancing

| Class | Full Name | Images (balanced) |
|---|---|---|
| akiec | Actinic Keratosis | 1,500 |
| bcc | Basal Cell Carcinoma | 1,500 |
| bkl | Benign Keratosis-like Lesions | 1,500 |
| df | Dermatofibroma | 1,500 |
| mel | Melanoma | 1,500 |
| nv | Melanocytic Nevi | 1,500 |
| vasc | Vascular Lesions | 1,500 |

---

## 🤖 Models

### Model 1 — Baseline (ResNet50)

| Property | Value |
|---|---|
| Backbone | ResNet50 |
| Dataset | Original imbalanced |
| Loss function | Categorical cross-entropy |
| Test accuracy | 79.1% |
| Macro F1-score | 0.78 |
| Parameters | 24.1M |
| Test samples | 487 |

### Model 2 — Proposed (EfficientNetB0) ✅ Deployed

| Property | Value |
|---|---|
| Backbone | EfficientNetB0 |
| Dataset | Balanced (oversampled) |
| Loss function | Focal loss (γ=2.0, α=0.25) |
| Test accuracy | **85.8%** |
| Macro F1-score | **0.86** |
| Parameters | **4.8M** |
| Test samples | 1,575 |

---

## 📁 Project Structure

```
Skin cancer apps/
│
├── appss.py                          # Main Flask application (EfficientNetB0)
├── app.py                            # Alternative Flask app (ResNet50)
│
├── best_model_balanced.keras         # Trained EfficientNetB0 model (best checkpoint)
├── best_efficientnetb0_final.h5      # EfficientNetB0 final saved model
├── Resnet50_best_final_model.h5      # Trained ResNet50 model
│
├── EfficientNetB0_Balanced (1).ipynb # Training notebook — EfficientNetB0
├── ResNet_Best (2).ipynb             # Training notebook — ResNet50
│
├── templates/
│   └── index.html                    # Main frontend (DermAI interface)
│
└── static/
    ├── ISIC_0024329.jpg              # Sample test images
    ├── ISIC_0024331.jpg
    ├── ISIC_0024432.jpg
    └── ISIC_0024515.jpg
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or 3.12
- pip package manager

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/skin-cancer-detection.git
cd skin-cancer-detection
```

### Step 2 — Install dependencies

```bash
pip install flask tensorflow pillow numpy
```

### Step 3 — Ensure model file is present

Make sure `best_model_balanced.keras` is in the root directory alongside `appss.py`. If the model file is too large for GitHub, download it separately and place it in the same folder.

---

## 🚀 Usage

### Run the application

```bash
python appss.py
```

### Open in browser

```
http://localhost:5000
```

### How to use

1. Open the browser at `http://localhost:5000`
2. Click **Upload Image** and select a dermoscopic skin lesion image
3. Click **Run Analysis**
4. View the predicted class and probability scores for all 7 lesion types

---

## 📈 Results

### Per-Class F1-Score Comparison

| Class | ResNet50 F1 | EfficientNetB0 F1 | Change |
|---|---|---|---|
| akiec | 0.69 | 0.74 | +0.05 |
| bcc | 0.86 | 0.94 | +0.08 |
| bkl | 0.76 | 0.75 | -0.01 |
| df | 0.60 | **1.00** | +0.40 ✅ |
| mel | 0.77 | 0.78 | +0.01 |
| nv | 0.83 | 0.81 | -0.02 |
| vasc | 0.93 | 0.99 | +0.06 |


---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep learning framework | TensorFlow / Keras |
| Model architecture | EfficientNetB0 (transfer learning) |
| Backend | Flask (Python) |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Image processing | Pillow (PIL) |
| Data handling | NumPy |
| Training environment | Jupyter Notebook (Anaconda) |
| Dataset | HAM10000 (Kaggle) |

---

## ⚠️ Important Notes

- This application is intended as a **decision-support tool only** and is **not a replacement** for professional dermatological diagnosis
- The model was trained and tested on the HAM10000 dataset and may not generalize to images taken in different clinical settings or with different imaging equipment
- For best results, use standard dermoscopic images similar to the HAM10000 dataset
- The model file `best_model_balanced.keras` must be loaded with `compile=False` due to the custom focal loss function used during training

---

## 📚 References

- A. Verma et al., "Skin Cancer Detection Using Deep Learning with EfficientNetB0," *ICAAAI 2025*.
- S. Moges, "Integration of Feature Fusion Strategy on EfficientNet for Skin Cancer Detection," M.Sc. thesis, Jimma University, 2023.
- S. Sumanth et al., "Skin Cancer Detection using EfficientNet," *IJERT*, vol. 14, no. 12, 2025.

---

## 👤 Author

**[      ]**
3rd Year Computer Science Student
Jinka University (JKU) — Department of Computer Science
May 2026

---

## 📄 License

This project is developed for academic purposes at Jimma University. All rights reserved.

---

> ⭐ If you find this project useful, please consider giving it a star on GitHub!

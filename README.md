# 🔍 Hybrid Image Forgery Detection System

## 📌 Overview

This project implements a **hybrid deep learning model** for detecting image forgeries using a combination of:

* Multiple deep learning backbones
* Handcrafted feature extraction
* Attention-based feature fusion

The system is trained on **CoMoFoD (copy-move)** and **CASIA v2.0 (splicing)** datasets to improve generalization across different forgery types.

---

## 🚀 Key Features

* Hybrid architecture:

  * ConvNeXt (local features)
  * EfficientNet (fine details)
  * Vision Transformer (global context)
* Handcrafted features:

  * DCT (frequency)
  * Zernike moments (shape)
  * LBP (texture)
* Attention-based feature fusion
* Handles dataset imbalance:

  * Focal Loss
  * Weighted sampling
* MixUp augmentation
* Ensemble evaluation (multi-model averaging)
* Automatic graph generation:

  * Loss
  * Accuracy
  * F1 Score
  * ROC Curve
  * Precision-Recall Curve
  * Confusion Matrix

---

## 🧠 Model Architecture

```text
Input Image
     ↓
 ┌────────────────────────────┐
 │ ConvNeXt   EfficientNet   │
 │ Vision Transformer (ViT)  │
 └────────────────────────────┘
               ↓
      Handcrafted Features
   (DCT + Zernike + LBP)
               ↓
        Feature Concatenation
               ↓
        Fusion Projection
               ↓
      Multi-head Attention
               ↓
        Residual Connection
               ↓
         Fully Connected
               ↓
      Forgery Prediction
```

---

## 📂 Datasets

### 🔹 CoMoFoD

* Copy-move forgery dataset
* Synthetic transformations
* Balanced using unique authentic samples

### 🔹 CASIA v2.0

* Splicing forgery dataset
* Real-world manipulation scenarios

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/forgery-detection.git
cd forgery-detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python train.py
```

### Training Techniques

* Focal Loss (handles imbalance)
* MixUp augmentation
* Layer-wise learning rates
* Mixed precision training
* Early stopping

### Outputs

* Model checkpoints → `checkpoints/`
* Graphs → `results/`

---

## 🧪 Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

---

## 📊 Results

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | ~93–94% |
| Precision | ~92%    |
| Recall    | ~95%    |
| F1 Score  | ~93–94% |

> Ensemble improves performance to ~95–96%

---

## 🧪 Ensemble Strategy

Final prediction uses:

```text
Average(best_model + intermediate_epoch_model)
```

Threshold is optimized using F1-score.

---

## 🌐 Backend (Flask API)

```bash
python app.py
```

### Endpoint

```
POST /predict
```

### Request

* Form-data: image file

### Response

```json
{
  "prediction": "FORGED",
  "confidence": 0.91
}
```

---

## 💻 Frontend (React)

```bash
cd frontend
npm install
npm start
```

Open:

```
http://localhost:3000
```

---

## 📊 Generated Graphs

Saved in `results/`:

* loss.png
* accuracy.png
* f1.png
* confusion_matrix.png
* roc.png
* pr.png
* bar.png

---

## ⚠️ Challenges

* Dataset imbalance (CoMoFoD vs CASIA)
* Domain gap (synthetic vs real images)
* Threshold sensitivity
* Feature distribution overlap

---

## 🔧 Improvements Applied

* Balanced sampling
* Focal loss
* Ensemble learning
* Threshold tuning
* Hybrid feature fusion

---

## 🔮 Future Work

* Forgery localization (pixel-level detection)
* Support for deepfake detection
* Lightweight model for real-time deployment
* FastAPI + Docker deployment
* Automatic threshold calibration

---

## 🛠️ Tech Stack

* Python
* PyTorch
* torchvision / timm
* NumPy, OpenCV, PIL
* Flask
* React.js
* Matplotlib, Seaborn

---

## 👨‍💻 Author

Manijitya Kumar Parimi

---

## 📜 License

MIT License

# speech-emotion-recognition
Classifying emotions from speech audio using MFCC features and a CNN, trained on the RAVDESS dataset.

<div align="center">

# 🎙️ Speech Emotion Recognition
### MFCC + CNN on the RAVDESS Dataset

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![librosa](https://img.shields.io/badge/librosa-audio-green?style=flat-square)](https://librosa.org)
[![Dataset](https://img.shields.io/badge/Dataset-RAVDESS-blueviolet?style=flat-square)](https://zenodo.org/record/1188976)
[![Accuracy](https://img.shields.io/badge/Accuracy-60%25-brightgreen?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

<br/>

> **Classifying human emotions from speech audio using deep learning.**  
> A personal research project by an undergraduate Data Science student at IIT Madras.

<br/>

</div>

---

## 📌 Overview

This project builds a **Speech Emotion Recognition (SER)** system that classifies audio recordings into one of 8 emotions — `angry`, `calm`, `happy`, `sad`, `fearful`, `disgust`, `surprised`, and `neutral` — using handcrafted acoustic features and a Convolutional Neural Network.

The goal was to engineer a clean, principled ML pipeline from scratch — not reproduce a tutorial. The journey involved two model collapses, discovering data leakage bugs, and diagnosing a 9.6M parameter explosion, all of which are documented below.

**Final result: 60% accuracy on 8-class classification (RAVDESS)**  
This is on par with the MFCC + SVM baseline tier reported in the SER literature.

---

## 🎯 Emotions Classified

| Code | Emotion | Code | Emotion |
|------|---------|------|---------|
| 01 | 😐 Neutral | 05 | 😠 Angry |
| 02 | 😌 Calm | 06 | 😨 Fearful |
| 03 | 😊 Happy | 07 | 🤢 Disgust |
| 04 | 😢 Sad | 08 | 😲 Surprised |

---

## 📊 Results

### Classification Report

| Emotion | Precision | Recall | F1-Score | Assessment |
|---------|-----------|--------|----------|------------|
| Angry | 0.91 | 0.74 | **0.82** | ✅ Excellent |
| Surprised | 0.85 | 0.73 | **0.79** | ✅ Strong |
| Disgust | 0.66 | 0.78 | **0.71** | ✅ Strong |
| Fearful | 0.62 | 0.72 | **0.67** | 🟡 Good |
| Happy | 0.88 | 0.44 | **0.59** | 🟡 Moderate |
| Sad | 0.31 | 0.85 | **0.46** | 🟡 Weak |
| Calm | 1.00 | 0.27 | **0.43** | 🔴 Weak |
| Neutral | 0.40 | 0.10 | **0.16** | 🔴 Difficult |
| **Overall** | **0.73** | **0.60** | **0.60** | **Solid Baseline** |

### Benchmark Comparison

| Approach | Typical Accuracy |
|----------|-----------------|
| MFCC + SVM (classic baseline) | 55–65% |
| **MFCC + CNN — this project** | **60%** |
| MFCC + delta + deeper CNN | 65–75% |
| Spectrogram + CNN + augmentation | 70–80% |
| Wav2Vec / HuBERT (transformer) | 85–90%+ |

---

## 🏗️ Model Architecture

```
Input: (120, 174, 1)  ← MFCC + delta + delta-delta, 174 time frames
│
├── Conv2D(32, 3×3) + BatchNorm + MaxPool(2×2)    →  (60, 87, 32)
├── Conv2D(64, 3×3) + BatchNorm + MaxPool(2×2)    →  (30, 43, 64)
├── Conv2D(128, 3×3) + BatchNorm + MaxPool(2×2)   →  (15, 21, 128)
│
├── GlobalAveragePooling2D                         →  (128,)
├── Dense(128, relu) + L2 regularisation
├── Dropout(0.4)
│
└── Dense(8, softmax)                              →  emotion class

Total parameters: 110,600 (434 KB)
```

> **Key design decision:** `GlobalAveragePooling2D` was used instead of `Flatten`.  
> Flatten on the tripled feature map produced **75,264 values → 9.6M Dense parameters** on a 1,440-sample dataset, causing model collapse. GlobalAveragePooling2D reduces this to **128 values → 16K parameters**.

---

## 🔬 Feature Extraction

Each `.wav` file is transformed into a **(120 × 174)** 2D feature matrix:

```python
mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)      # spectral envelope
delta  = librosa.feature.delta(mfcc)                            # velocity
delta2 = librosa.feature.delta(mfcc, order=2)                   # acceleration
features = np.concatenate([mfcc, delta, delta2], axis=0)        # (120, 174)
```

- **40 MFCC coefficients** — how the human ear perceives spectral shape
- **40 delta coefficients** — how energy changes frame to frame
- **40 delta-delta coefficients** — rate of change of the delta

---

## 🗺️ Development Journey

This project involved 5 distinct phases, including two model collapses:

| Phase | Accuracy | Status | Key Change |
|-------|----------|--------|------------|
| v1 — Baseline | 25–30% | ⚠️ Class bias | Basic MFCC + CNN, no tuning |
| v2 — First Fixes | ~10% | ❌ Collapse | Aggressive class weights caused collapse |
| v3 — Stabilisation | ~60% | ✅ Stable | Fixed normalisation leakage, simplified model |
| v4 — Delta Features | 30% | ❌ Collapse | Flatten + 120-channel input = 9.6M params |
| **v5 — Final** | **60%** | ✅ Solid | GlobalAveragePooling2D + 3 Conv blocks + L2 |

### Key bugs discovered and fixed

**1. Data leakage in normalisation**
```python
# ❌ Wrong — uses test statistics in training
X = (X - np.mean(X)) / (np.std(X) + 1e-6)   # computed before split!

# ✅ Correct — train stats only
mean, std = X_train.mean(), X_train.std() + 1e-6
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std
```

**2. Augmentation applied to test data**
```python
# ❌ Wrong — augments both train and test
def extract_features(file_path):
    audio = audio + 0.003 * np.random.randn(len(audio))   # inside extraction!

# ✅ Correct — separate function, called on X_train only after split
def augment_train(X_train):
    return X_train + 0.003 * np.random.randn(*X_train.shape)
```

**3. Hardcoded class weights → data-driven**
```python
# ❌ Wrong — arbitrary
class_weights[neutral_index] = 2.5

# ✅ Correct — computed from actual label distribution
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/parkhi-12-code/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the RAVDESS dataset
Download from [Zenodo](https://zenodo.org/record/1188976) and place the audio files in:
```
data/
└── Ravdess/
    ├── Actor_01/
    ├── Actor_02/
    └── ...
```

### 4. Run
```bash
python ser_model.py
```

Outputs: confusion matrix (`confusion_matrix.png`), training curves (`training_curves.png`), and a classification report in the terminal.

---

## 📁 Project Structure

```
speech-emotion-recognition/
│
├── ser_model.py          ← main training script
├── requirements.txt      ← dependencies
├── README.md             ← this file
│
├── data/
│   └── Ravdess/          ← dataset (not tracked by git)
│
└── confusion_matrix.png  ← evaluation output
```

---

## 🔧 Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.12 |
| Audio Processing | `librosa` |
| Deep Learning | `TensorFlow 2 / Keras` |
| ML Utilities | `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn` |
| Version Control | Git + GitHub |

---

## 🔭 Future Improvements

| Improvement | Expected Impact |
|-------------|----------------|
| Pitch shift + time stretch augmentation | +3–5% accuracy, reduces train/val gap |
| ZCR + RMS energy features | Better separation of calm/neutral/sad |
| t-SNE visualisation of MFCC embeddings | Research-quality cluster analysis |
| Cross-corpus evaluation (EMODB, CREMA-D) | Measure generalisation |
| BiGRU layer replacing GlobalAvgPool | +5–8% accuracy, captures temporal sequence |

---

## 📖 Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

- 24 professional actors (12 male, 12 female)
- 8 emotion categories
- ~1,440 `.wav` audio files
- Sample rate: 22,050 Hz

> Livingstone SR, Russo FA (2018) *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*. PLoS ONE 13(5): e0196391.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Parkhi Yadav**  
B.Sc. Data Science — IIT Madras  
Personal Research Project

</div>

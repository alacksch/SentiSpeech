# 🎤 SentiSpeech: Speech Emotion Recognition using SVM

**SentiSpeech** is a machine learning project that detects and classifies human emotions from speech using audio feature extraction and a Support Vector Machine (SVM) classifier. Built on datasets like **RAVDESS**, **CREMA-D** and **TESS**, the system analyzes vocal characteristics such as pitch, tone, and energy to recognize emotions like **neutral**, **happy**, **sad**, and **angry**.

---

## 📁 Datasets

This project supports multiple popular emotional speech datasets:

- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
  - [RAVDESS on Zenodo](https://zenodo.org/record/1188976)  
  - [RAVDESS on Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

- **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)  
  - [CREMA-D on Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad)

- **TESS** (Toronto emotional speech set)  
  - [TESS on Kaggle](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)

---

## 🔍 Features

- 🎧 Audio loading and preprocessing with Librosa  
- 📊 Feature extraction: MFCCs, Chroma, Spectral Contrast, Pitch, RMS, Zero Crossing Rate, Spectral Bandwidth, and more  
- 🧠 Emotion parsing from structured filenames and metadata  
- 🧪 Model training pipeline with StandardScaler, Linear Discriminant Analysis (LDA), and Support Vector Machine (SVM)  
- 🔄 Hyperparameter tuning using Grid Search with cross-validation  
- 📈 Good for generalization for many languages because of how many audio files are provided and how different they all are.

---

## 🧠 Emotions Detected

- 😐 Neutral  
- 😊 Happy  
- 😢 Sad  
- 😠 Angry  

---

## 🚀 Getting Started

1. Prepare datasets directories for RAVDESS, CREMA-D, TESS, and/or MSP-IMPROV  
2. Run the data preparation and feature extraction pipeline  
3. Train the model using model.py
4. Predict using the saved model

---

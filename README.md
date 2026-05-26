# Speech Emotion Recognition Using CNN

**Domain:** AI / Machine Learning · Audio Signal Processing
**Stack:** Python · TensorFlow · Keras · MFCC · Mel Spectrogram · Flask
**Results:** 98.32% training accuracy · 94.0% test accuracy
**Status:** Complete

---

## Overview

A convolutional neural network (CNN) pipeline for classifying human emotions from speech audio. The system extracts MFCC and Mel Spectrogram features from raw audio, trains a CNN classifier, and deploys the trained model as a Flask inference API.

---

## Problem Statement

Speech emotion recognition has applications in mental health monitoring, human-computer interaction, and call centre analytics. The challenge is extracting discriminative features from raw audio and learning a classifier that generalises beyond the training distribution. This project evaluates CNN architectures on MFCC and Mel Spectrogram representations.

---

## Features

- MFCC and Mel Spectrogram feature extraction pipeline
- CNN classifier trained end-to-end on extracted features
- Data augmentation: time shifting, pitch shifting, noise injection
- Training/validation/test split with reproducible random seed
- Model evaluation: accuracy, confusion matrix, per-class F1
- Flask REST API for inference on new audio samples
- Model serialisation and loading

---

## Technical Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Deep Learning | TensorFlow 2.x · Keras |
| Feature Extraction | librosa (MFCC, Mel Spectrogram) |
| Data Handling | NumPy · pandas |
| Evaluation | scikit-learn (confusion matrix, F1) |
| Visualisation | matplotlib · seaborn |
| Deployment | Flask |

---

## Architecture

```
[Raw Audio (.wav)] ──► [Preprocessing] ──► [Feature Extraction]
                                                    │
                                        ┌───────────┴───────────┐
                                    [MFCC]               [Mel Spectrogram]
                                        └───────────┬───────────┘
                                              [CNN Model]
                                                    │
                                         [Softmax Classifier]
                                                    │
                                         [Emotion Label Output]
```

**CNN Architecture:**
- Input: (n_mfcc x time_frames x 1) feature map
- Conv2D → BatchNorm → ReLU → MaxPool (x3 blocks)
- Flatten → Dense(256) → Dropout(0.4) → Dense(n_classes)
- Output: Softmax over emotion classes

---

## Feature Extraction

**MFCC:** 40 coefficients extracted per frame, 512-sample FFT window, 50% overlap. Delta and delta-delta features appended to capture temporal dynamics.

**Mel Spectrogram:** 128 Mel filter banks, normalised log-scale. Used as a 2D image input to the CNN.

**Data Augmentation:**
- Time shift: +/-10% of signal length
- Gaussian noise injection: SNR ~20dB
- Pitch shift: +/-2 semitones

---

## Results

| Metric | Value |
|---|---|
| Training Accuracy | 98.32% |
| Test Accuracy | 94.0% |
| Macro F1 Score | 0.93 |

Confusion matrix and per-class results in `/docs/evaluation/`

---

## Flask Inference API

```
POST /predict
Content-Type: multipart/form-data
Body: audio file (.wav)

Response: {"emotion": "happy", "confidence": 0.94}
```

---

## Repository Structure

```
speech-emotion-recognition-cnn/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   ├── app.py
│   └── utils.py
├── models/
│   └── emotion_cnn.h5
├── docs/
│   ├── architecture.png
│   └── evaluation/
│       ├── confusion_matrix.png
│       └── training_curves.png
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/smaneesh031/speech-emotion-recognition-cnn
cd speech-emotion-recognition-cnn
pip install -r requirements.txt
python src/train.py
python api/app.py
```

---

## Future Improvements

- Evaluate transformer-based (wav2vec) feature extraction
- Extend to real-time streaming inference
- Implement speaker-independent cross-validation
- Add attention mechanism to CNN for interpretability

# Engine Health Prognostics Using a Hybrid CNN–BiLSTM Network with Attention Mechanism

A PyTorch implementation of a hybrid deep learning framework for Remaining Useful Life (RUL) prediction of aircraft turbofan engines using the NASA CMAPSS dataset.

---

## Abstract

Predicting the Remaining Useful Life (RUL) of complex industrial systems is a critical task in predictive maintenance. This project presents a hybrid deep learning framework that combines Convolutional Neural Networks (CNNs), Bidirectional Long Short-Term Memory (BiLSTM) networks, and an attention mechanism to estimate engine health from multivariate sensor measurements. The proposed framework utilizes soft RUL formulation, delta sensor feature engineering, and sliding-window sequence generation to effectively model both local temporal characteristics and long-term degradation behaviour. Experimental evaluation on the NASA CMAPSS FD001 and FD003 datasets demonstrates strong predictive performance for aircraft engine health prognostics.

---

# Highlights

- Hybrid CNN–BiLSTM architecture
- Attention mechanism for temporal feature weighting
- Soft Remaining Useful Life (RUL) formulation
- Delta sensor feature engineering
- Sliding-window sequence generation
- PyTorch implementation
- Evaluation on NASA CMAPSS FD001 and FD003 datasets

---

# Model Architecture

```
NASA CMAPSS Dataset
        │
        ▼
Data Preprocessing
(Standardization)
        │
        ▼
Delta Feature Engineering
        │
        ▼
Sliding Window Generation
        │
        ▼
1D CNN Feature Extraction
        │
        ▼
BiLSTM
        │
        ▼
Attention Mechanism
        │
        ▼
Fully Connected Layer
        │
        ▼
Remaining Useful Life Prediction
```

---

# Dataset

The model is trained and evaluated using the **NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS)** dataset.

The experiments reported in the paper use:

- FD001
- FD003

Dataset Link:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

---

# Methodology

The proposed framework consists of the following stages:

1. Data preprocessing
2. Sensor normalization
3. Delta sensor feature extraction
4. Soft RUL target generation
5. Sliding window sequence creation
6. CNN-based local feature extraction
7. BiLSTM temporal modelling
8. Attention-based feature aggregation
9. Remaining Useful Life prediction

---


## Performance

The proposed CNN–BiLSTM–Attention framework was evaluated on the NASA CMAPSS benchmark datasets.

| Dataset | RMSE | MAE | MAPE (%) | R² |
|:--------|-----:|----:|---------:|---:|
| FD001 | **8.08** | **6.07** | **8.91** | **0.934** |
| FD003 | **8.53** | **5.79** | **9.19** | **0.930** |

Additional evaluation metrics include:

- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

---

# Repository Structure

```
predictive-maintenance-rul-prediction/
│
├── train.py
├── requirements.txt
├── README.md
├── architecture.png
├── results/
└── .gitignore
```

---

# Requirements

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- tqdm

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

# Citation

If you use this repository, please cite:

> **Engine Health Prognostics Using a Hybrid CNN–BiLSTM Network with Attention Mechanism**

*(Citation details will be updated after publication.)*

---

# Author

**Upasana Dwivedy**

Manipal University Jaipur


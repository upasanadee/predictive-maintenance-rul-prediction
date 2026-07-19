# Predictive Maintenance for Aircraft Turbofan Engines

A deep learning framework for Remaining Useful Life (RUL) prediction of aircraft turbofan engines using the NASA CMAPSS dataset. The proposed model combines CNN-based feature extraction, parallel BiLSTM and BiGRU networks, and an attention mechanism to effectively capture both local sensor patterns and long-term temporal dependencies.

## Features

- CNN feature extractor for sensor data
- Parallel BiLSTM and BiGRU architecture
- Attention mechanism for temporal feature aggregation
- Soft Remaining Useful Life (RUL) labeling
- Delta sensor feature engineering
- Hybrid MSE + MAE loss function
- Implemented using PyTorch

## Dataset

This project uses the **NASA CMAPSS** turbofan engine degradation dataset.

Dataset:
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Dataset subset used:
- FD002

## Model Pipeline

Sensor Data
→ Standardization
→ Delta Feature Generation
→ Sliding Window Creation
→ CNN Feature Extraction
→ Parallel BiLSTM + BiGRU
→ Attention Layer
→ Fully Connected Layer
→ Remaining Useful Life Prediction

## Results

| Metric | Value |
|---------|------:|
| RMSE | XX.XX |
| MAE | XX.XX |
| MAPE | XX.XX% |
| R² | 0.XXX |

*(Replace the values with your final experimental results.)*

## Repository Structure

```
train.py           # Training and evaluation pipeline
requirements.txt   # Python dependencies
README.md
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## Author

Upasana Dwivedy

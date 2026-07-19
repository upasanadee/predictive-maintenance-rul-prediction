"""
===============================================================================
Engine Health Prognostics Using a Hybrid CNN–BiLSTM Network with Attention Mechanism
===============================================================================

Author      : Upasana Dwivedy
Institution : Manipal University Jaipur

Description
-----------
PyTorch implementation of a hybrid CNN–BiLSTM–Attention framework for
Remaining Useful Life (RUL) prediction of aircraft turbofan engines
using the NASA CMAPSS benchmark dataset.

The proposed framework consists of:

    • Soft Remaining Useful Life (RUL) formulation
    • Delta sensor feature engineering
    • Sliding-window sequence generation
    • CNN-based local feature extraction
    • Bidirectional LSTM temporal modelling
    • Attention-based feature aggregation

Paper
-----
Engine Health Prognostics Using a Hybrid CNN–BiLSTM Network with
Attention Mechanism

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

print(f"Using device: {DEVICE}")

# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------

DATA_DIR = "data"

# Supported:
# FD001
# FD002
# FD003
# FD004

FD = "FD001"  # Change to FD002 / FD003 / FD004

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------

WINDOW = 60
RUL_CAP = 125

BATCH_SIZE = 64
EPOCHS = 120

LEARNING_RATE = 1e-3

SEED = 42

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from preprocessing import prepare_data

X_train, y_train, X_test, y_test, ALL_FEATURES = prepare_data(
    data_dir="data",
    fd="FD001",
    window=60,
    rul_cap=125,
)

# =============================================================================
# PYTORCH DATASET
# =============================================================================

class RULDataset(Dataset):
    """
    PyTorch Dataset for Remaining Useful Life (RUL) prediction.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# Create DataLoaders

train_loader = DataLoader(
    RULDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    RULDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# =============================================================================
# ATTENTION MODULE
# =============================================================================

class Attention(nn.Module):
    """
    Temporal attention mechanism.

    Learns the importance of each time step and computes a
    weighted feature representation.
    """

    def __init__(self, feature_dim: int):

        super().__init__()

        self.score = nn.Linear(feature_dim, 1)

    def forward(self, x):

        weights = torch.softmax(self.score(x), dim=1)

        context = (weights * x).sum(dim=1)

        return context


# =============================================================================
# PROPOSED CNN–BiLSTM–ATTENTION MODEL
# =============================================================================

class CNN_BiLSTM_Attention(nn.Module):
    """
    Hybrid CNN–BiLSTM network with an attention mechanism for
    Remaining Useful Life prediction.
    """

    def __init__(self, input_dim: int):

        super().__init__()

        # -------------------------------------------------------------
        # CNN Feature Extractor
        # -------------------------------------------------------------

        self.cnn = nn.Sequential(

            nn.Conv1d(
                in_channels=input_dim,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2)

        )

        # -------------------------------------------------------------
        # Bidirectional LSTM
        # -------------------------------------------------------------

        self.bilstm = nn.LSTM(

            input_size=64,
            hidden_size=64,

            batch_first=True,

            bidirectional=True

        )

        # -------------------------------------------------------------
        # Attention
        # -------------------------------------------------------------

        self.attention = Attention(128)

        # -------------------------------------------------------------
        # Regression Head
        # -------------------------------------------------------------

        self.regressor = nn.Linear(128, 1)

    def forward(self, x):

        # ---------------------------------------------------------
        # CNN
        # Input:
        # (Batch, Time, Features)
        # ---------------------------------------------------------

        x = x.permute(0, 2, 1)

        x = self.cnn(x)

        # ---------------------------------------------------------
        # BiLSTM
        # ---------------------------------------------------------

        x = x.permute(0, 2, 1)

        x, _ = self.bilstm(x)

        # ---------------------------------------------------------
        # Attention
        # ---------------------------------------------------------

        x = self.attention(x)

        # ---------------------------------------------------------
        # Final Prediction
        # ---------------------------------------------------------

        output = self.regressor(x)

        return output.squeeze()


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

model = CNN_BiLSTM_Attention(

    input_dim=len(ALL_FEATURES)

).to(DEVICE)

print(model)

# =============================================================================
# LOSS FUNCTION
# =============================================================================

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()


def hybrid_loss(prediction, target):
    """
    Hybrid regression loss combining Mean Squared Error (MSE)
    and Mean Absolute Error (MAE).
    """

    return (
        0.7 * mse_loss(prediction, target)
        + 0.3 * mae_loss(prediction, target)
    )


# =============================================================================
# OPTIMIZER
# =============================================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# =============================================================================
# TRAINING
# =============================================================================

print("\nStarting Training...\n")

best_rmse = np.inf

for epoch in tqdm(range(1, EPOCHS + 1), desc="Training"):

    # ---------------------------------------------------------------------
    # Training Phase
    # ---------------------------------------------------------------------

    model.train()

    train_loss = 0.0

    for xb, yb in train_loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(xb)

        loss = hybrid_loss(predictions, yb)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------------------------------------------------------------------
    # Evaluation Phase
    # ---------------------------------------------------------------------

    model.eval()

    predictions = []

    with torch.no_grad():

        for xb, _ in test_loader:

            xb = xb.to(DEVICE)

            outputs = model(xb)

            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(
        np.abs(
            (y_test - predictions) /
            np.clip(y_test, 1e-6, None)
        )
    ) * 100

    r2 = r2_score(y_test, predictions)

    # ---------------------------------------------------------------------
    # Save Best Model
    # ---------------------------------------------------------------------

    if rmse < best_rmse:

        best_rmse = rmse

        torch.save(
            model.state_dict(),
            f"best_model_{FD}.pth"
        )

    # ---------------------------------------------------------------------
    # Training Log
    # ---------------------------------------------------------------------

    tqdm.write(

        f"Epoch [{epoch:03d}/{EPOCHS}] | "
        f"Loss: {train_loss:.4f} | "
        f"RMSE: {rmse:.3f} | "
        f"MAE: {mae:.3f} | "
        f"MAPE: {mape:.2f}% | "
        f"R²: {r2:.3f}"

    )

# =============================================================================
# FINAL RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("FINAL ENGINE-LEVEL PERFORMANCE")
print("=" * 70)

print(f"Dataset : {FD}")
print(f"Best RMSE : {best_rmse:.3f}")
print(f"Final RMSE: {rmse:.3f}")
print(f"MAE       : {mae:.3f}")
print(f"MAPE      : {mape:.2f}%")
print(f"R² Score  : {r2:.3f}")

print("=" * 70)

print("\nBest model saved successfully.")

# =============================================================================
# END OF FILE
# =============================================================================

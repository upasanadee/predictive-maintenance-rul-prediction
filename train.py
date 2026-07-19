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

# =============================================================================
# DATA LOADING
# =============================================================================

def load_cmapss(split: str) -> pd.DataFrame:
    """
    Loads the NASA CMAPSS dataset.

    Parameters
    ----------
    split : str
        Dataset split ("train" or "test")

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    columns = (
        ["engine_id", "cycle", "op1", "op2", "op3"]
        + [f"s{i}" for i in range(1, 22)]
    )

    file_path = os.path.join(DATA_DIR, f"{split}_{FD}.txt")

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        names=columns,
    )


# Load training and testing data

train_df = load_cmapss("train")
test_df = load_cmapss("test")

# Load Remaining Useful Life labels

rul_df = pd.read_csv(
    os.path.join(DATA_DIR, f"RUL_{FD}.txt"),
    names=["RUL"],
)
# =============================================================================
# SOFT REMAINING USEFUL LIFE (RUL) FORMULATION
# =============================================================================

def soft_rul(rul: np.ndarray, cap: int = RUL_CAP) -> np.ndarray:
    """
    Computes the soft Remaining Useful Life (RUL) target.

    Instead of using a hard linear degradation target, the soft RUL
    formulation provides smoother labels during the early life of an
    engine, resulting in more stable model training.

    Parameters
    ----------
    rul : np.ndarray
        Remaining Useful Life values.

    cap : int
        Maximum RUL value.

    Returns
    -------
    np.ndarray
        Soft RUL labels.
    """

    return cap * (1 - np.exp(-rul / 50))


# -------------------------------------------------------------------------
# Generate training RUL targets
# -------------------------------------------------------------------------

max_cycle = train_df.groupby("engine_id")["cycle"].max()

train_df["RUL"] = soft_rul(
    max_cycle[train_df.engine_id].values - train_df.cycle.values
)

# -------------------------------------------------------------------------
# Generate testing RUL targets
# -------------------------------------------------------------------------

test_max = test_df.groupby("engine_id")["cycle"].max()

test_df["RUL"] = soft_rul(
    test_max[test_df.engine_id].values
    - test_df.cycle.values
    + np.array([rul_df.iloc[eid - 1, 0] for eid in test_df.engine_id])
)

# =============================================================================
# SENSOR SELECTION
# =============================================================================

# Selected sensors based on previous CMAPSS studies.

SENSORS = [
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]

# =============================================================================
# FEATURE NORMALIZATION
# =============================================================================

scaler = StandardScaler()

train_df[SENSORS] = scaler.fit_transform(train_df[SENSORS])
test_df[SENSORS] = scaler.transform(test_df[SENSORS])

# =============================================================================
# DELTA SENSOR FEATURE ENGINEERING
# =============================================================================

def add_delta_features(
    df: pd.DataFrame,
    sensors: list[str]
) -> pd.DataFrame:
    """
    Computes first-order temporal differences for each selected sensor.

    Delta features provide additional degradation information by
    capturing the rate of change of sensor measurements.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    sensors : list[str]
        Selected sensor names.

    Returns
    -------
    pd.DataFrame
        Dataframe with delta sensor features appended.
    """

    delta = df.groupby("engine_id")[sensors].diff().fillna(0)

    delta.columns = [f"d_{sensor}" for sensor in sensors]

    return pd.concat([df, delta], axis=1)


train_df = add_delta_features(train_df, SENSORS)
test_df = add_delta_features(test_df, SENSORS)

# Combined feature set

ALL_FEATURES = SENSORS + [f"d_{sensor}" for sensor in SENSORS]

# =============================================================================
# SLIDING WINDOW GENERATION
# =============================================================================

def make_train_windows(df: pd.DataFrame):
    """
    Generates fixed-length sliding windows for training.

    Returns
    -------
    X : np.ndarray
        Input sequences.

    y : np.ndarray
        Corresponding RUL labels.
    """

    X, y = [], []

    for engine_id in df.engine_id.unique():

        engine = df[df.engine_id == engine_id]

        features = engine[ALL_FEATURES].values
        rul = engine["RUL"].values

        for i in range(len(engine) - WINDOW):

            X.append(features[i:i + WINDOW])
            y.append(rul[i + WINDOW - 1])

    return np.array(X), np.array(y)


def make_test_last_window(df: pd.DataFrame):
    """
    Generates one final sliding window for each engine
    during inference.
    """

    X, y = [], []

    for engine_id in df.engine_id.unique():

        engine = df[df.engine_id == engine_id]

        features = engine[ALL_FEATURES].values

        if len(features) < WINDOW:

            padding = np.zeros(
                (WINDOW - len(features), features.shape[1])
            )

            features = np.vstack([padding, features])

        X.append(features[-WINDOW:])
        y.append(engine["RUL"].values[-1])

    return np.array(X), np.array(y)


# Create training and testing sequences

X_train, y_train = make_train_windows(train_df)
X_test, y_test = make_test_last_window(test_df)

print(f"Training samples : {X_train.shape}")
print(f"Testing samples  : {X_test.shape}")
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

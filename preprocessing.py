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

import pandas as pd
from .config import DATA_RAW, FEATURE_COLUMNS, SENSOR_COLUMNS, RUL_CAP, FAILURE_THRESHOLD

NASA_COLUMNS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + SENSOR_COLUMNS
REQUIRED_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]

def load_training_frame() -> pd.DataFrame:
    path = DATA_RAW / "train_FD001.txt"
    if not path.exists():
        raise FileNotFoundError(
            "NASA FD001 data not found. Run `python -m src.download_nasa_data` first."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None, names=NASA_COLUMNS)
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df["rul"] = df["engine_id"].map(max_cycles) - df["cycle"]
    df["rul_capped"] = df["rul"].clip(upper=RUL_CAP)
    df["failure_soon"] = (df["rul"] <= FAILURE_THRESHOLD).astype(int)
    return df

def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing feature columns: {missing}")
    return df[FEATURE_COLUMNS].astype(float)

def make_prediction_row(values: dict) -> pd.DataFrame:
    return pd.DataFrame([{c: float(values[c]) for c in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)

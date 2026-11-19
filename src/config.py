from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:7001")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
FAILURE_MODEL_NAME = "predictive-maintenance-failure-classifier"
RUL_MODEL_NAME = "predictive-maintenance-rul-regressor"
RUL_CAP = 125
FAILURE_THRESHOLD = 30
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
FEATURE_COLUMNS = ["cycle", "setting_1", "setting_2", "setting_3"] + SENSOR_COLUMNS

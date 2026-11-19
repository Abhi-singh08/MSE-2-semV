from contextlib import asynccontextmanager
import os
import subprocess
import sys
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .config import (
    MLFLOW_TRACKING_URI, FAILURE_MODEL_NAME, RUL_MODEL_NAME,
    MODELS_DIR, FEATURE_COLUMNS
)
from .data import make_prediction_row

models = {}

def load_models():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        models["failure"] = mlflow.sklearn.load_model(f"models:/{FAILURE_MODEL_NAME}@champion",skops_trusted_types=["sklearn.tree._tree.Tree"])
        models["rul"] = mlflow.sklearn.load_model(f"models:/{RUL_MODEL_NAME}@champion",skops_trusted_types=["sklearn.tree._tree.Tree"])
    except Exception:
        # Local artifacts are a fallback for first-time development.
        required = [MODELS_DIR / "failure_classifier.joblib", MODELS_DIR / "rul_regressor.joblib", MODELS_DIR / "anomaly_detector.joblib"]
        if not all(p.exists() for p in required):
            raise RuntimeError("Models not found. Start MLflow and run `python -m src.train` first.")
        models["failure"] = joblib.load(required[0])
        models["rul"] = joblib.load(required[1])
    anomaly_path = MODELS_DIR / "anomaly_detector.joblib"
    if anomaly_path.exists():
        models["anomaly"] = joblib.load(anomaly_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_models()
    except Exception as exc:
        # Allow API startup before training; endpoints report model-not-ready.
        models.clear()
        app.state.model_load_error = str(exc)
    yield

app = FastAPI(title="Predictive Maintenance API", version="1.0.0", lifespan=lifespan)

class SensorInput(BaseModel):
    cycle: float = Field(gt=0)
    setting_1: float
    setting_2: float
    setting_3: float
    s1: float; s2: float; s3: float; s4: float; s5: float; s6: float; s7: float
    s8: float; s9: float; s10: float; s11: float; s12: float; s13: float; s14: float
    s15: float; s16: float; s17: float; s18: float; s19: float; s20: float; s21: float

@app.get("/")
def root():
    return {"status": "ready", "application": "Predictive Maintenance"}

@app.get("/health")
def health():
    return {"status": "healthy" if "failure" in models and "rul" in models else "model_not_ready",
            "loaded_models": list(models.keys())}

@app.post("/predict")
def predict(data: SensorInput):
    if "failure" not in models or "rul" not in models:
        raise HTTPException(status_code=503, detail="Models are not loaded. Train models and reload.")
    row = make_prediction_row(data.model_dump())
    failure = int(models["failure"].predict(row)[0])
    rul = max(0.0, float(models["rul"].predict(row)[0]))
    anomaly = None
    if "anomaly" in models:
        anomaly = int(models["anomaly"].predict(row)[0] == -1)
    return {
        "failure_soon": failure,
        "risk": "HIGH" if failure else "LOW",
        "estimated_rul_cycles": round(rul, 2),
        "is_anomaly": anomaly
    }

@app.post("/reload-model")
def reload_model():
    try:
        load_models()
        return {"status": "success", "message": "Models reloaded"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/retrain")
def retrain():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.train"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True, text=True, timeout=1800
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-3000:] or result.stdout[-3000:])
        load_models()
        return {"status": "success", "message": "Retraining completed and models reloaded",
                "training_output": result.stdout[-2000:]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}")

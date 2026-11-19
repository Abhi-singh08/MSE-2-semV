import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from .config import (
    MLFLOW_TRACKING_URI, MODELS_DIR, FAILURE_MODEL_NAME, RUL_MODEL_NAME, RUL_CAP
)
from .data import load_training_frame, feature_frame

MIN_FAILURE_F1 = 0.50
RANDOM_STATE = 42

def register_alias(model_name: str, run_id: str, alias: str = "champion"):
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"name='{model_name}'")
    version = next((v for v in versions if v.run_id == run_id), None)
    if version is None:
        raise RuntimeError(f"Could not find registered version for {model_name}, run {run_id}")
    client.set_registered_model_alias(model_name, alias, version.version)

def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("nasa-cmapss-predictive-maintenance")
    df = load_training_frame()
    X = feature_frame(df)
    y_cls = df["failure_soon"].astype(int)
    y_rul = df["rul_capped"].astype(float)

    # Split by engine, not by individual rows, to prevent the same engine
    # appearing in both training and validation sets.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y_cls, groups=df["engine_id"]))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    yc_train, yc_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
    yr_train, yr_test = y_rul.iloc[train_idx], y_rul.iloc[test_idx]

    classifier = RandomForestClassifier(
        n_estimators=300, max_depth=14, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    regressor = RandomForestRegressor(
        n_estimators=300, max_depth=16, random_state=RANDOM_STATE, n_jobs=-1
    )
    anomaly_model = IsolationForest(contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "dataset": "NASA C-MAPSS FD001",
            "split_strategy": "GroupShuffleSplit by engine_id",
            "algorithm": "RandomForest",
            "n_estimators": 300,
            "max_depth_classifier": 14,
            "max_depth_regressor": 16,
            "failure_threshold_cycles": 30,
            "rul_cap": RUL_CAP,
            "validation_engine_fraction": 0.25,
            "random_state": RANDOM_STATE
        })
        classifier.fit(X_train, yc_train)
        regressor.fit(X_train, yr_train)
        anomaly_model.fit(X_train)

        pred_cls = classifier.predict(X_test)
        pred_rul = regressor.predict(X_test)
        accuracy = accuracy_score(yc_test, pred_cls)
        precision = precision_score(yc_test, pred_cls, zero_division=0)
        recall = recall_score(yc_test, pred_cls, zero_division=0)
        f1 = f1_score(yc_test, pred_cls, zero_division=0)
        mae = mean_absolute_error(yr_test, pred_rul)
        rmse = float(np.sqrt(mean_squared_error(yr_test, pred_rul)))
        metrics = {
            "failure_accuracy": float(accuracy),
            "failure_precision": float(precision),
            "failure_recall": float(recall),
            "failure_f1": float(f1),
            "rul_mae": float(mae),
            "rul_rmse": float(rmse)
        }
        mlflow.log_metrics(metrics)

        # Quality gate: keep the previous champion if the candidate fails.
        if f1 < MIN_FAILURE_F1:
            raise ValueError(
                f"Candidate rejected: failure F1={f1:.3f} is below "
                f"the configured threshold {MIN_FAILURE_F1:.2f}. Existing champion is unchanged."
            )

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier, MODELS_DIR / "failure_classifier.joblib")
        joblib.dump(regressor, MODELS_DIR / "rul_regressor.joblib")
        joblib.dump(anomaly_model, MODELS_DIR / "anomaly_detector.joblib")

        mlflow.sklearn.log_model(
            sk_model=classifier, name="failure-classifier",
            registered_model_name=FAILURE_MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        mlflow.sklearn.log_model(
            sk_model=regressor, name="rul-regressor",
            registered_model_name=RUL_MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        run_id = run.info.run_id

    register_alias(FAILURE_MODEL_NAME, run_id)
    register_alias(RUL_MODEL_NAME, run_id)
    print("Validation metrics:")
    print(json.dumps(metrics, indent=2))
    print("Candidate passed quality gate. MLflow champion aliases updated.")
    return metrics

if __name__ == "__main__":
    train()

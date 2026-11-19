# AI-Powered Predictive Maintenance System using ML and MLOps

This project uses the **real NASA C-MAPSS Turbofan Engine Degradation Simulation dataset (FD001)**. No synthetic/demo dataset is used.

## What is included
- NASA FD001 downloader (official NASA Prognostics Data Repository archive)
- Data preparation and Remaining Useful Life (RUL) label generation
- Random Forest failure-soon classifier
- Random Forest RUL regressor
- Isolation Forest anomaly detector
- Engine-wise validation split to avoid putting rows from the same engine in both train and validation sets
- MLflow experiment tracking, model registration and `champion` aliases
- FastAPI prediction and controlled retraining endpoints
- Streamlit dashboard
- Pytest API tests
- Docker and Docker Compose
- GitHub Actions CI and GHCR publishing

No Kubernetes, Argo CD, Airflow, DVC, Prometheus, Grafana or Evidently are included, per the agreed scope.

## Requirements
Windows 10/11, Python 3.11 recommended, VS Code. Docker Desktop is needed only for the container stage.

## 1. Create environment (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, use `python -m venv .venv`.

## 2. Download the actual NASA C-MAPSS FD001 dataset

```powershell
python -m src.download_nasa_data
```

This downloads NASA's official Turbofan Engine Degradation Simulation archive and extracts only:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

The source archive is about 12 MB. Internet access is required for this one-time download. If the NASA host is temporarily unavailable, download the archive manually from the official NASA repository linked below and place those three files in `data/raw/`.

Official source: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Dataset citation: A. Saxena and K. Goebel (2008), “Turbofan Engine Degradation Simulation Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

## 3. Prepare and inspect the data

```powershell
python -m src.prepare_nasa_data
```

The training file contains engine IDs, cycle numbers, 3 operating settings and 21 sensor readings. Training RUL is derived from each engine's final observed cycle. The RUL target is capped at 125 cycles, and the failure-soon classification label is 1 when RUL is 30 cycles or less.

## 4. Start MLflow (Terminal 2)

Keep this terminal running:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 7001
```

MLflow UI: http://127.0.0.1:7001

## 5. Train and register models (Terminal 1)

```powershell
python -m src.train
```

This trains the classifier, RUL regressor and anomaly detector. Validation is split by engine ID (not random rows) to reduce data leakage. MLflow logs parameters/metrics and registers the classifier and regressor. A candidate only replaces the `champion` alias if it passes the configured F1 quality gate.

## 6. Start FastAPI (Terminal 3)

```powershell
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## 7. Start Streamlit (Terminal 4)

```powershell
streamlit run app/streamlit_app.py
```

Dashboard: http://localhost:8501

The dashboard accepts one row of cycle, operating settings and 21 sensor values and displays failure-soon risk, estimated RUL and anomaly status.

## 8. Controlled retraining

Use the Streamlit sidebar button **Run controlled retraining**, or call:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/retrain
```

Retraining uses the latest training file currently in `data/raw/train_FD001.txt`. It does not invent new data or retrain on prediction requests. If the dataset is unchanged, the training data is unchanged too. The candidate is evaluated before promotion; a rejected candidate does not replace the previous champion alias.

## 9. Tests

```powershell
python -m pytest -v
```

## 10. Docker Compose (after local flow works)

```powershell
docker compose up --build
```

- MLflow: http://localhost:7001
- FastAPI: http://localhost:8000/docs
- Streamlit: http://localhost:8501

## Structure

```text
predictive-maintenance-mlops/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/                  # downloaded NASA FD001 files (gitignored)
│   └── processed/
├── models/
├── src/
│   ├── api.py
│   ├── config.py
│   ├── data.py
│   ├── download_nasa_data.py
│   ├── prepare_nasa_data.py
│   └── train.py
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

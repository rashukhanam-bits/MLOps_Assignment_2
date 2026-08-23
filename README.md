# End-to-end MLOps pipeline

Use case : Binary image classification (Cats vs Dogs) for a pet adoption platform.

---

# 1 - Prerequisites

Install:

```bash
python3 --version      # need 3.10+
docker --version       # Docker Desktop or Engine
git --version
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Directory Layout:

Directory layout:

```
MLOps_Assignmnet_2/
├── data/                      # raw/processed data (DVC-tracked, not in git)
├── src/
│   ├── data/
│   │   ├── download_data.py   # pulls Kaggle Cats vs Dogs dataset
│   │   └── preprocess.py      # resize/RGB/split/augment
│   ├── models/
│   │   ├── model.py           # CNN architecture
│   │   └── train.py           # training + MLflow logging
│   └── api/
│       ├── app.py             # FastAPI inference service
│       ├── schemas.py         # request/response models
│       └── inference.py       # model loading + predict utility
├── tests/
│   ├── test_preprocess.py
│   └── test_inference.py
├── docker/
│   └── Dockerfile
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/
│   ├── ci.yml                 # M3: test + build + push image
│   └── cd.yml                 # M4: deploy + smoke test
├── monitoring/
│   └── metrics.py             # Prometheus counters/latency
├── scripts/
│   └── smoke_test.sh          # M4: post-deploy health/predict check
├── docker-compose.yml
├── requirements.txt
├── dvc.yaml
├── params.yaml
├── .dvcignore
└── README.md
```
---

# 2 - M1: Model Development & Experiment Tracking

## Task 1 - Git for code versioning and DVC for data versioning

```bash
git add .
git commit -m "Initial commit"
git push
```

```bash
pip install dvc
dvc init
```

Download the data set from https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

Copy it to data/raw

Track it with DVC:

```bash
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track raw dataset with DVC"
```

Set up a DVC remote

```bash
dvc remote add -d localstorage ../dvc-storage
dvc push
```
Preprocess (resize to 224x224 RGB, split 80/10/10, augment) and version the output too:

```bash
python src/data/preprocess.py
dvc add data/processed
git add dvc.yaml dvc.lock params.yaml
git commit -m "Add preprocessing pipeline output"
dvc push
```

## Task 2 and Task 3 - Model Building and Experiment Tracking

```bash
mlflow ui --port 5000 
```

Train:

```bash
python src/models/train.py --epochs 10 --batch-size 32 --lr 0.001
```

Open `http://localhost:5000` to browse runs.

# 3 - M2: Model Packaging & Containerization

## Task1 - Inference Service

`src/api/app.py` is a FastAPI app with:
- `GET /health` — liveness/readiness check
- `POST /predict` — accepts a base64 image (or file upload), returns `{label, probability}`

Run it locally:

```bash
uvicorn src.api.app:app --reload --port 8000
```

Test:

```bash
curl http://localhost:8000/health

 curl.exe -X POST http://localhost:8000/predict -F "file=@data/processed/test/cats/6.jpg"  
```

## Task2 - Environment Specification

`requirements.txt` pins every ML library version (tensorflow, fastapi, uvicorn,
mlflow, dvc, pillow, numpy, pytest, prometheus-client, etc.) for reproducibility.

## Task3 - Containerization

```bash
docker build -t cats-dogs-api:local -f docker/Dockerfile .
docker run -p 8000:8000 cats-dogs-api:local
curl http://localhost:8000/health
curl.exe -X POST http://localhost:8000/predict -F "file=@data/processed/test/cats/6.jpg"  
```

# 4 - M3: CI Pipeline for Build, Test & Image Creation

## Task1 - Automated Testing

Tests are in tests folder.

```bash
python -m pytest tests/ -v
```

## Task2 and Task3- CI Setup and Artifact Publishing

`.github/workflows/ci.yml` runs on every push / PR to `main`:
1. Checks out repo
2. Sets up Python, installs `requirements.txt`
3. Runs `pytest tests/ -v`
4. Builds the Docker image
5. Logs into Docker Hub (via `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repo secrets)
6. Pushes the image as `docker.io/<you>/cats-dogs-api:<git-sha>` and `:latest`

Set the two secrets in **GitHub → Settings → Secrets and variables → Actions**
before pushing.

# 5 - M4: CD Pipeline & Deployment


Using Docker:

```bash
docker compose up -d
bash scripts/smoke_test.sh http://localhost:8000
```

`.github/workflows/cd.yml` runs after CI succeeds on `main`: it pulls the new
image, updates the Deployment `docker compose pull
&& docker compose up -d`, then runs `scripts/smoke_test.sh` against the live
endpoint — the job **fails the pipeline** if the smoke test fails, per spec.

# 6 - M5: Monitoring, Logs & Final Submission

### 5.1 Logging & metrics

`src/api/app.py` logs every request (timestamp, endpoint, status, latency —
no image bytes/PII) via Python `logging`, and exposes Prometheus-format
counters/histograms at `GET /metrics` (request count, latency histogram) using
`monitoring/metrics.py`.

### 5.2 Post-deployment performance tracking

```bash
python scripts/simulate_traffic.py --n 50 --endpoint http://localhost:8000
```

This sends a batch of test images with known true labels, records predictions,
and writes `reports/post_deploy_eval.csv` + prints accuracy — simulating
production drift monitoring.

### 5.3 Final packaging

```bash
zip -r submission.zip . -x ".venv/*" -x "data/raw/*" -x "data/processed/*" -x "mlruns/*" -x ".git/*"
```
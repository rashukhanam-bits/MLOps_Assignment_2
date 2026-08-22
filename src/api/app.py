"""
FastAPI inference service for the Cats vs Dogs classifier.

Endpoints:
    GET  /health   - liveness/readiness check
    POST /predict  - accepts an image file, returns predicted label + probabilities
    GET  /metrics  - Prometheus metrics (request count, latency)

Run:
    uvicorn src.api.app:app --reload --port 8000
"""
import logging
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.api.inference import predict, get_model, MODEL_PATH
from src.api.schemas import HealthResponse, PredictionResponse
from monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY, PREDICTION_COUNT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("cats-dogs-api")

app = FastAPI(title="Cats vs Dogs Inference API", version="1.0.0")


@app.on_event("startup")
def load_model_on_startup():
    try:
        get_model()
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except FileNotFoundError as e:
        # Don't crash the container if the model isn't baked in yet; /health will report it.
        logger.warning("Model not loaded at startup: %s", e)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        get_model()
        loaded = True
    except FileNotFoundError:
        loaded = False
    REQUEST_COUNT.labels(endpoint="/health", status="200").inc()
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    start = time.time()
    try:
        image_bytes = await file.read()
        result = predict(image_bytes)
    except FileNotFoundError as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="503").inc()
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="400").inc()
        logger.warning("Bad prediction request: %s", e)
        raise HTTPException(status_code=400, detail="Invalid image input")

    latency = time.time() - start
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
    REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
    PREDICTION_COUNT.labels(predicted_class=result["label"]).inc()

    # Log request metadata only -- never raw image bytes (no PII/sensitive data)
    logger.info(
        "predict filename=%s label=%s probability=%.4f latency_ms=%.1f",
        file.filename, result["label"], result["probability"], latency * 1000,
    )
    return PredictionResponse(**result)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

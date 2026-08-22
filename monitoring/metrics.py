"""Prometheus metrics for the inference service (M5: monitoring)."""
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of requests received",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made, by predicted class",
    ["predicted_class"],
)

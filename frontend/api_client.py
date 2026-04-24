"""HTTP client for FastAPI backend."""
import httpx
import streamlit as st

BASE_URL = "http://localhost:8000"


def _client():
    return httpx.Client(base_url=BASE_URL, timeout=120)


def health_check():
    try:
        with _client() as c:
            return c.get("/health").json()
    except Exception:
        return {"status": "offline", "model_ready": False}


def upload_and_train(file_bytes: bytes, filename: str):
    with _client() as c:
        r = c.post("/upload-and-train",
                   files={"file": (filename, file_bytes, "application/octet-stream")})
        r.raise_for_status()
        return r.json()


def upload_predict(file_bytes: bytes, filename: str):
    with _client() as c:
        r = c.post("/upload-predict",
                   files={"file": (filename, file_bytes, "application/octet-stream")})
        r.raise_for_status()
        return r.json()


def predict_single(payload: dict):
    with _client() as c:
        r = c.post("/predict", json=payload)
        r.raise_for_status()
        return r.json()


def get_advice(payload: dict):
    with _client() as c:
        r = c.post("/advise", json=payload)
        r.raise_for_status()
        return r.json()


def calc_downtime(payload: dict):
    with _client() as c:
        r = c.post("/downtime-calculator", json=payload)
        r.raise_for_status()
        return r.json()


def get_shap(file_bytes: bytes, filename: str):
    with _client() as c:
        r = c.post("/shap",
                   files={"file": (filename, file_bytes, "application/octet-stream")})
        r.raise_for_status()
        return r.json()


def get_audit_logs(limit: int = 100):
    with _client() as c:
        r = c.get(f"/audit-logs?limit={limit}")
        r.raise_for_status()
        return r.json()


def get_health_score(payload: dict):
    with _client() as c:
        r = c.post("/health-score", json=payload)
        r.raise_for_status()
        return r.json()


def get_ttf(file_bytes: bytes, filename: str):
    with _client() as c:
        r = c.post("/ttf", files={"file": (filename, file_bytes, "application/octet-stream")})
        r.raise_for_status()
        return r.json()


def get_model_registry():
    with _client() as c:
        r = c.get("/model-registry")
        r.raise_for_status()
        return r.json()

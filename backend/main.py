"""FastAPI application – REST + WebSocket."""
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import logging

from backend import ml_engine, alerts, llm_advisor, iot_simulator, file_parser, downtime_calculator
from backend.database import insert_audit_log, insert_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PredMaint API",
    description="Industrial Machine Predictive Maintenance System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": ml_engine.model_exists()}


# ── File Upload & Training ────────────────────────────────────────────────────

@app.post("/upload-and-train")
async def upload_and_train(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = file_parser.parse_upload(file.filename, content)
    except ValueError as e:
        raise HTTPException(400, str(e))

    schema = file_parser.infer_schema(df)
    ok, missing = file_parser.validate_columns(df, ml_engine.FEATURE_COLS + [ml_engine.TARGET])
    if not ok:
        raise HTTPException(422, f"Missing required columns: {missing}")

    metrics = ml_engine.train_model(df)
    await insert_audit_log("model_trained", "dataset", file.filename,
                           {"rows": len(df), "metrics": metrics})
    return {"schema": schema, "metrics": metrics, "message": "Model trained successfully"}


@app.post("/upload-predict")
async def upload_predict(file: UploadFile = File(...)):
    """Upload a file and get batch predictions."""
    if not ml_engine.model_exists():
        raise HTTPException(400, "Model not trained yet. Call /upload-and-train first.")
    content = await file.read()
    try:
        df = file_parser.parse_upload(file.filename, content)
    except ValueError as e:
        raise HTTPException(400, str(e))

    result_df = ml_engine.predict_batch(df)
    await insert_audit_log("batch_predict", "dataset", file.filename, {"rows": len(df)})
    return {
        "total": len(result_df),
        "breakdowns_predicted": int(result_df["prediction"].sum()),
        "critical": int((result_df["risk_level"] == "CRITICAL").sum()),
        "results": result_df[["asset_tag", "machine_type", "prediction",
                               "probability", "risk_level", "anomaly_score"]
                              if "asset_tag" in result_df.columns
                              else ["prediction", "probability", "risk_level", "anomaly_score"]
                              ].head(500).to_dict(orient="records"),
    }


# ── Single Prediction ─────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    asset_tag: Optional[str] = "UNKNOWN"
    machine_type: Optional[str] = "CNC Lathe"
    temp_bearing_degC: float
    temp_motor_degC: float
    vibration_h_mms: float
    vibration_v_mms: float
    oil_pressure_bar: float
    load_pct: float
    shaft_rpm: float
    power_consumption_kw: float


@app.post("/predict")
async def predict(reading: SensorReading, request: Request):
    if not ml_engine.model_exists():
        raise HTTPException(400, "Model not trained yet.")
    data = reading.model_dump()
    result = ml_engine.predict_single(data)
    data.update(result)

    await insert_prediction({
        "asset_tag": data["asset_tag"],
        "machine_type": data["machine_type"],
        "prediction": result["prediction"],
        "probability": result["probability"],
        "features": {k: data[k] for k in ml_engine.FEATURE_COLS if k in data},
    })
    await insert_audit_log("predict", "machine", data["asset_tag"],
                           result, ip=request.client.host if request.client else None)

    if result["risk_level"] in ("HIGH", "CRITICAL"):
        await alerts.fire_breakdown_alert(
            data["asset_tag"], data["machine_type"],
            result["probability"], result["risk_level"]
        )

    return data


# ── SHAP ──────────────────────────────────────────────────────────────────────

@app.post("/shap")
async def shap_analysis(file: UploadFile = File(...)):
    if not ml_engine.model_exists():
        raise HTTPException(400, "Model not trained yet.")
    content = await file.read()
    df = file_parser.parse_upload(file.filename, content)
    result = ml_engine.compute_shap(df)
    return result


# ── LLM Advisor ───────────────────────────────────────────────────────────────

@app.post("/advise")
async def advise(reading: SensorReading):
    if not ml_engine.model_exists():
        raise HTTPException(400, "Model not trained yet.")
    data = reading.model_dump()
    result = ml_engine.predict_single(data)
    data.update(result)

    # Top SHAP features (simplified)
    data["top_features"] = "temp_bearing_degC, vibration_h_mms, power_consumption_kw"
    advice = await llm_advisor.get_maintenance_advice(data)
    await insert_audit_log("llm_advise", "machine", data["asset_tag"], {"risk": result["risk_level"]})
    return {"prediction": result, "advice": advice}


# ── Downtime Calculator ───────────────────────────────────────────────────────

class DowntimeRequest(BaseModel):
    machine_type: str = "CNC Lathe"
    mttr_hours: Optional[float] = None
    hourly_production_inr: Optional[float] = None
    repair_cost_inr: Optional[float] = None
    annual_breakdowns: int = 12
    pdm_system_cost_inr: float = 500000


@app.post("/downtime-calculator")
async def calc_downtime(req: DowntimeRequest):
    result = downtime_calculator.calculate_downtime(
        machine_type=req.machine_type,
        mttr_hours=req.mttr_hours,
        hourly_production_inr=req.hourly_production_inr,
        repair_cost_inr=req.repair_cost_inr,
        annual_breakdowns=req.annual_breakdowns,
        pdm_system_cost_inr=req.pdm_system_cost_inr,
    )
    return result.__dict__


# ── Audit Logs ────────────────────────────────────────────────────────────────

@app.get("/audit-logs")
async def get_audit_logs(limit: int = 100):
    from backend.database import get_supabase
    db = get_supabase()
    if db is None:
        return {"logs": [], "message": "Supabase not configured"}
    try:
        resp = db.table("audit_logs").select("*").order("created_at", desc=True).limit(limit).execute()
        return {"logs": resp.data}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── WebSocket: Live IoT Simulation ────────────────────────────────────────────

@app.websocket("/ws/live-sensors")
async def live_sensors(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected for live sensors")
    try:
        await iot_simulator.stream_sensor_data(websocket, interval=1.0)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

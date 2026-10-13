"""
FastAPI application – REST + WebSocket
High-throughput backend: all ML logic runs here.
Streamlit frontend is a pure display layer.
"""
import asyncio
import json
import logging
import os
from typing import Optional
from backend.timescaledb import init_pool, close_pool, insert_audit_log, insert_prediction
from backend.ot_writeback import OPCUAWriteBack

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from backend import ml_engine, alerts, file_parser, downtime_calculator
from backend import health_score, ttf_predictor, model_registry
from backend.copilot.response_schema import (
    ExplainableWorkOrderResponse,
    build_explainable_response,
)
# (Removed Supabase import; using TimescaleDB)
from backend.sensor_validator import validate_reading
from backend.ingestion import parse_mqtt_payload, parse_opcua_frame, ingest
from backend.iot_simulator import StressTestHarness, StressMode, MACHINES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PredMaint API",
    description="Industrial Machine Predictive Maintenance — High-Throughput Backend",
    version="2.0.0",
)

# Global write‑back handler (configured via env vars)
_opcua_writeback = OPCUAWriteBack(
    endpoint=os.getenv("OPCUA_ENDPOINT", "opc.tcp://localhost:4840"),
    node_id=os.getenv("OPCUA_EMERGENCY_NODE", "ns=2;s=Emergency_Stop_Trigger"),
    auth={"username": os.getenv("OPCUA_USER"), "password": os.getenv("OPCUA_PASSWORD")}
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared harness instance ───────────────────────────────────────────────────
_harness = StressTestHarness(mode=StressMode.DEGRADING)
_prev_readings: dict = {}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_ready": ml_engine.model_exists(),
        "version": "2.0.0",
        "protocols": ["REST", "WebSocket", "MQTT-ingestion", "OPC-UA-ingestion"],
    }


# ── Training ──────────────────────────────────────────────────────────────────

@app.post("/upload-and-train")
async def upload_and_train(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = file_parser.parse_upload(file.filename, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    ok, missing = file_parser.validate_columns(df, ml_engine.FEATURE_COLS + [ml_engine.TARGET])
    if not ok:
        raise HTTPException(422, f"Missing required columns: {missing}")
    metrics = ml_engine.train_model(df)
    await insert_audit_log("model_trained", "dataset", file.filename,
                           {"rows": len(df), "metrics": metrics})
    return {"schema": file_parser.infer_schema(df), "metrics": metrics}


@app.post("/upload-predict")
async def upload_predict(file: UploadFile = File(...)):
    if not ml_engine.model_exists():
        raise HTTPException(400, "Model not trained yet.")
    content = await file.read()
    try:
        df = file_parser.parse_upload(file.filename, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    result_df = ml_engine.predict_batch(df)
    await insert_audit_log("batch_predict", "dataset", file.filename, {"rows": len(df)})
    cols = (["asset_tag", "machine_type", "prediction", "probability", "risk_level", "anomaly_score"]
            if "asset_tag" in result_df.columns
            else ["prediction", "probability", "risk_level", "anomaly_score"])
    return {
        "total": len(result_df),
        "breakdowns_predicted": int(result_df["prediction"].sum()),
        "critical": int((result_df["risk_level"] == "CRITICAL").sum()),
        "results": result_df[cols].head(500).to_dict(orient="records"),
    }


# ── Single Prediction ─────────────────────────────────────────────────────────

class SensorReadingIn(BaseModel):
    asset_tag: Optional[str] = "UNKNOWN"
    machine_type: Optional[str] = "CNC Lathe"
    # Optional spectrum-derived peak; reported only when the gateway supplies it.
    fft_peak_hz: Optional[float] = None
    temp_bearing_degC: float
    temp_motor_degC: float
    vibration_h_mms: float
    vibration_v_mms: float
    oil_pressure_bar: float
    load_pct: float
    shaft_rpm: float
    power_consumption_kw: float


@app.post("/predict", response_model=ExplainableWorkOrderResponse)
async def predict(reading: SensorReadingIn, request: Request):
    if not ml_engine.model_exists():
        raise HTTPException(400, "Model not trained yet.")
    data = reading.model_dump()

    # Validate before predicting
    val = validate_reading(data, _prev_readings.get(data["asset_tag"]))
    if val.malfunction:
        return build_explainable_response(
            data,
            prediction={"validation_status": "MALFUNCTION", "risk_level": "SENSOR_MALFUNCTION"},
        )

    _prev_readings[data["asset_tag"]] = data
    result = ml_engine.predict_single(data)
    data.update(result)
    data["validation_status"] = "OK"

    await insert_prediction({
        "asset_tag": data["asset_tag"], "machine_type": data["machine_type"],
        "prediction": result["prediction"], "probability": result["probability"],
        "features": {k: data[k] for k in ml_engine.FEATURE_COLS if k in data},
    })
    await insert_audit_log("predict", "machine", data["asset_tag"],
                           result, ip=request.client.host if request.client else None)
    if result["risk_level"] in ("HIGH", "CRITICAL"):
        await alerts.fire_breakdown_alert(
            data["asset_tag"], data["machine_type"],
            result["probability"], result["risk_level"])
    shap_result = None
    copilot_result = None
    if result.get("risk_level") in ("HIGH", "CRITICAL", "MEDIUM") or result.get("probability", 0) >= 0.5:
        from backend.copilot import run_maintenance_copilot
        copilot_result = run_maintenance_copilot(data, trigger_source="rest_api_predict")
        shap_result = copilot_result.get("shap_top_features")
    if shap_result is None:
        from backend.copilot.shap_explainer import explain_reading_shap
        shap_result = explain_reading_shap(data).get("top_features", [])
    return build_explainable_response(
        data,
        prediction=result,
        copilot_result=copilot_result,
        shap_features=shap_result,
    )


# ── MQTT Ingestion ────────────────────────────────────────────────────────────

class MQTTPayload(BaseModel):
    topic: str
    payload: dict
    machine_type: Optional[str] = "CNC Lathe"


@app.post("/ingest/mqtt")
async def ingest_mqtt(msg: MQTTPayload):
    """
    Ingest a raw MQTT message. Normalises payload, validates sensors,
    runs ML prediction. Returns pre-calculated scores.
    """
    asset_tag = msg.topic.split("/")[2] if len(msg.topic.split("/")) >= 3 else "UNKNOWN"
    normalised = parse_mqtt_payload(
        msg.topic, msg.payload, msg.machine_type,
        _prev_readings.get(asset_tag))

    if normalised.get("malfunction"):
        return {**normalised, "prediction": None, "probability": None,
                "risk_level": "SENSOR_MALFUNCTION"}

    if ml_engine.model_exists():
        try:
            result = ml_engine.predict_single(normalised)
            normalised.update(result)
        except Exception as e:
            logger.error(f"Prediction failed for MQTT reading: {e}")

    _prev_readings[asset_tag] = normalised
    return normalised


# ── OPC-UA Ingestion ──────────────────────────────────────────────────────────

class OPCUAFrame(BaseModel):
    asset_tag: str
    machine_type: Optional[str] = "CNC Lathe"
    nodes: list[dict]


@app.post("/ingest/opcua")
async def ingest_opcua(frame: OPCUAFrame):
    """
    Ingest an OPC-UA data frame (list of NodeId/Value pairs).
    Validates and returns pre-calculated ML scores.
    """
    normalised = parse_opcua_frame(
        frame.nodes, frame.asset_tag, frame.machine_type,
        _prev_readings.get(frame.asset_tag))

    if normalised.get("malfunction"):
        return {**normalised, "prediction": None, "probability": None,
                "risk_level": "SENSOR_MALFUNCTION"}

    if ml_engine.model_exists():
        try:
            result = ml_engine.predict_single(normalised)
            normalised.update(result)
        except Exception as e:
            logger.error(f"Prediction failed for OPC-UA reading: {e}")

    _prev_readings[frame.asset_tag] = normalised
    return normalised


# ── SHAP ──────────────────────────────────────────────────────────────────────

@app.post("/shap")
async def shap_analysis(file: UploadFile = File(...)):
    if not ml_engine.model_exists():
        raise HTTPException(400, "Model not trained yet.")
    content = await file.read()
    df = file_parser.parse_upload(file.filename, content)
    return ml_engine.compute_shap(df)


# ── LLM Advisor ───────────────────────────────────────────────────────────────

@app.post("/advise", response_model=ExplainableWorkOrderResponse)
async def advise(reading: SensorReadingIn):
    if not ml_engine.model_exists():
        raise HTTPException(400, "Model not trained yet.")
    data = reading.model_dump()
    validation = validate_reading(data, _prev_readings.get(data["asset_tag"]))
    if validation.malfunction:
        return build_explainable_response(
            data,
            prediction={"validation_status": "MALFUNCTION", "risk_level": "SENSOR_MALFUNCTION"},
        )
    result = ml_engine.predict_single(data)
    data.update(result)
    from backend.copilot import run_maintenance_copilot
    copilot_result = run_maintenance_copilot(data, trigger_source="rest_api_advise")
    await insert_audit_log("copilot_advise", "machine", data["asset_tag"],
                           {"risk": result["risk_level"], "source": "grounded_copilot"})
    return build_explainable_response(
        data,
        prediction=result,
        copilot_result=copilot_result,
    )


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
        machine_type=req.machine_type, mttr_hours=req.mttr_hours,
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
        resp = (db.table("audit_logs").select("*")
                .order("created_at", desc=True).limit(limit).execute())
        return {"logs": resp.data}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Health Score ──────────────────────────────────────────────────────────────

@app.post("/health-score")
async def get_health_score(reading: SensorReadingIn):
    result = health_score.compute_health_score(reading.model_dump())
    return result.__dict__


@app.post("/fleet-health")
async def get_fleet_health(file: UploadFile = File(...)):
    content = await file.read()
    df = file_parser.parse_upload(file.filename, content)
    result = health_score.compute_fleet_health(df)
    return result.to_dict(orient="records")


# ── TTF ───────────────────────────────────────────────────────────────────────

@app.post("/ttf")
async def time_to_failure(file: UploadFile = File(...)):
    content = await file.read()
    df = file_parser.parse_upload(file.filename, content)
    if "asset_tag" not in df.columns:
        raise HTTPException(422, "Dataset must have 'asset_tag' column")
    results = ttf_predictor.fleet_ttf(df)
    await insert_audit_log("ttf_analysis", "dataset", file.filename,
                           {"assets": len(results)})
    return {"results": results}


# ── Model Registry ────────────────────────────────────────────────────────────

@app.get("/model-registry")
async def get_model_registry():
    registry = model_registry.get_registry()
    comparison = model_registry.compare_models()
    return {
        "registry": registry,
        "comparison": comparison.to_dict(orient="records") if not comparison.empty else [],
        "active_version": model_registry.get_active_version(),
    }


# ── Maintenance Copilot Endpoints ─────────────────────────────────────────────

class CopilotQueryRequest(BaseModel):
    machine_type: str = "CNC Lathe"
    query: str
    top_k: int = 3
    min_threshold: Optional[float] = 0.52


@app.post("/copilot/diagnose", response_model=ExplainableWorkOrderResponse)
async def copilot_diagnose(reading: SensorReadingIn):
    """
    Run the LangGraph Maintenance Copilot on sensor telemetry.
    Computes SHAP, retrieves SOP manuals via RAG, evaluates novelty,
    and drafts a confidence-gated work order.
    """
    from backend.copilot import run_maintenance_copilot
    data = reading.model_dump()
    validation = validate_reading(data, _prev_readings.get(data["asset_tag"]))
    if validation.malfunction:
        return build_explainable_response(
            data,
            prediction={"validation_status": "MALFUNCTION", "risk_level": "SENSOR_MALFUNCTION"},
        )
    result = run_maintenance_copilot(data, trigger_source="rest_api_diagnose")
    await insert_audit_log(
        "copilot_work_order_generated",
        "machine",
        data["asset_tag"],
        {
            "work_order_id": result.get("work_order_id"),
            "confidence_score": result.get("confidence_score"),
            "escalation_tier": result.get("escalation_tier"),
            "is_novel": result.get("is_novel_anomaly"),
        }
    )
    return build_explainable_response(
        data,
        prediction={
            "risk_level": result.get("risk_level"),
            "probability": result.get("breakdown_probability"),
        },
        copilot_result=result,
    )


@app.post("/copilot/search-manuals")
async def copilot_search_manuals(req: CopilotQueryRequest):
    """Search indexed equipment manuals & SOPs using vector RAG retriever."""
    from backend.copilot import get_rag_indexer
    indexer = get_rag_indexer()
    return indexer.search_manuals(
        machine_type=req.machine_type,
        query=req.query,
        top_k=req.top_k,
        min_threshold=req.min_threshold or 0.52,
    )


@app.get("/copilot/manuals")
async def copilot_list_manuals(machine_type: Optional[str] = None):
    """List available equipment SOP manuals."""
    from backend.copilot import get_all_manuals, get_manuals_for_machine
    if machine_type:
        return get_manuals_for_machine(machine_type)
    return get_all_manuals()


@app.get("/copilot/work-orders")
async def copilot_get_work_orders():
    """Retrieve history of generated work orders."""
    from backend.copilot import WORK_ORDERS_HISTORY
    return {"work_orders": WORK_ORDERS_HISTORY}


# ── WebSocket: Pre-calculated scores pushed from backend ─────────────────────

@app.websocket("/ws/live-sensors")
async def live_sensors_raw(websocket: WebSocket):
    """Raw sensor stream (backward compatible)."""
    await websocket.accept()
    logger.info("WS /ws/live-sensors connected")
    try:
        from backend.iot_simulator import stream_sensor_data
        await stream_sensor_data(websocket, interval=1.0)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/scored-stream")
async def scored_stream(websocket: WebSocket):
    """
    High-throughput WebSocket that pushes pre-calculated scores.
    Streamlit connects here — receives anomaly scores, health indices,
    TTF estimates, and automatic Copilot triage when anomalies occur.
    """
    await websocket.accept()
    logger.info("WS /ws/scored-stream connected")

    try:
        from backend.iot_simulator import generate_all_machines_reading
        from backend.health_score import compute_health_score
        from backend.copilot import run_maintenance_copilot
        import asyncio

        while True:
            readings = generate_all_machines_reading()
            scored_payload = []

            for r in readings:
                # Sensor validation
                val = validate_reading(r, _prev_readings.get(r["asset_tag"]))
                _prev_readings[r["asset_tag"]] = r

                # ML Scoring
                if ml_engine.model_exists() and not val.malfunction:
                    try:
                        pred_res = ml_engine.predict_single(r)
                        r.update(pred_res)
                    except Exception as e:
                        logger.warning(f"Live scoring error: {e}")

                # Health Score
                hs = compute_health_score(r)
                r["health_score"] = hs.health_score
                r["health_status"] = hs.status
                r["validation_status"] = "MALFUNCTION" if val.malfunction else "OK"

                # Check for anomaly / health score drop to trigger Copilot
                if (r.get("health_score", 100) < 70 or r.get("risk_level") in ("HIGH", "CRITICAL")) and not val.malfunction:
                    # Autonomous Copilot analysis
                    try:
                        copilot_res = run_maintenance_copilot(r, trigger_source="ws_scored_stream")
                        r["copilot_triggered"] = True
                        r["copilot_work_order"] = copilot_res.get("work_order_structured")
                    except Exception as ce:
                        logger.error(f"WS Copilot execution failed: {ce}")

                scored_payload.append(r)

            await websocket.send_text(json.dumps({
                "timestamp": datetime.datetime.now().isoformat() if hasattr(datetime, "datetime") else str(pd.Timestamp.now()),
                "data": scored_payload,
            }))
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.info("WS /ws/scored-stream disconnected")
    except Exception as e:
        logger.error(f"WS /ws/scored-stream error: {e}")


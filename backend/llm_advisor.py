"""LLM-powered prescriptive maintenance advisor."""
import logging
from backend.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert industrial maintenance engineer AI.
Given sensor readings and a breakdown prediction, provide:
1. Root cause analysis (2-3 likely causes)
2. Immediate actions (within 24 hours)
3. Preventive maintenance schedule
4. Spare parts to stock
5. Estimated downtime if ignored (hours)
Be concise, practical, and use engineering terminology."""


def _build_user_prompt(data: dict) -> str:
    return f"""
Machine: {data.get('machine_type', 'Unknown')} | Asset: {data.get('asset_tag', 'N/A')}
Breakdown Probability: {data.get('probability', 0):.1%} | Risk: {data.get('risk_level', 'N/A')}

Sensor Readings:
- Bearing Temp: {data.get('temp_bearing_degC', 'N/A')} °C
- Motor Temp: {data.get('temp_motor_degC', 'N/A')} °C
- Horizontal Vibration: {data.get('vibration_h_mms', 'N/A')} mm/s
- Vertical Vibration: {data.get('vibration_v_mms', 'N/A')} mm/s
- Oil Pressure: {data.get('oil_pressure_bar', 'N/A')} bar
- Load: {data.get('load_pct', 'N/A')} %
- Shaft RPM: {data.get('shaft_rpm', 'N/A')}
- Power: {data.get('power_consumption_kw', 'N/A')} kW
- Anomaly Score: {data.get('anomaly_score', 'N/A')}

Top SHAP features driving this prediction: {data.get('top_features', 'N/A')}

Provide prescriptive maintenance recommendations.
"""


async def get_maintenance_advice(data: dict) -> str:
    s = get_settings()
    prompt = _build_user_prompt(data)

    if s.llm_provider == "groq" and s.groq_api_key:
        return await _call_groq(prompt, s.groq_api_key)
    elif s.openai_api_key:
        return await _call_openai(prompt, s.openai_api_key)
    else:
        return _fallback_advice(data)


async def _call_openai(prompt: str, api_key: str) -> str:
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        return _fallback_advice({})


async def _call_groq(prompt: str, api_key: str) -> str:
    try:
        import httpx
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 600,
            "temperature": 0.3,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        return _fallback_advice({})


def _fallback_advice(data: dict) -> str:
    risk = data.get("risk_level", "HIGH")
    return f"""## Prescriptive Maintenance Recommendation (Rule-Based)

**Risk Level: {risk}**

### Likely Root Causes
1. Bearing wear or lubrication failure (elevated temperature + vibration)
2. Misalignment or imbalance in rotating components
3. Hydraulic seal degradation (if oil pressure anomaly detected)

### Immediate Actions (Next 24h)
- Inspect bearing condition and lubrication levels
- Check shaft alignment with dial indicator
- Verify oil pressure and check for leaks
- Review recent maintenance history

### Preventive Schedule
- Daily: Visual inspection + temperature check
- Weekly: Vibration measurement + oil level check
- Monthly: Full bearing inspection + alignment check
- Quarterly: Oil change + seal inspection

### Spare Parts to Stock
- Bearing set (matched pair)
- Oil seals and O-rings
- Coupling elements

### Estimated Downtime if Ignored
- 8–24 hours unplanned downtime within 7 days

*Configure OpenAI/Groq API key for AI-powered recommendations.*
"""

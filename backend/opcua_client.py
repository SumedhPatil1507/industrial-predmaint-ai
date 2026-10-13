"""Safety-gated asynchronous OPC-UA PLC write-back.

This module has no connection or task side effects at import time. Automated
commands are derived only from validated telemetry and ``SafetyThresholds``;
LLM text is accepted for audit context but never parsed as an authorization.
"""
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from asyncua import Client
from asyncua.ua import VariantType
from opentelemetry.trace import Status, StatusCode

from backend.sensor_validator import validate_reading
from backend.telemetry import get_tracer

LOGGER = logging.getLogger(__name__)
tracer = get_tracer(__name__)
_SAFE_ASSET_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class InterlockAction(str, Enum):
    NONE = "none"
    THROTTLE = "throttle"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class SafetyThresholds:
    """Physical limits applied to validated readings (engineering units)."""

    emergency_vibration_mms: float = 12.0
    throttle_vibration_mms: float = 8.0
    emergency_bearing_temp_c: float = 110.0
    throttle_bearing_temp_c: float = 95.0
    emergency_pressure_bar: float = 1.0
    throttle_pressure_bar: float = 2.5
    throttle_speed_fraction: float = 0.5

    def __post_init__(self) -> None:
        positive = (
            self.emergency_vibration_mms,
            self.throttle_vibration_mms,
            self.emergency_bearing_temp_c,
            self.throttle_bearing_temp_c,
            self.emergency_pressure_bar,
            self.throttle_pressure_bar,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("safety thresholds must be finite positive values")
        if self.throttle_speed_fraction <= 0 or self.throttle_speed_fraction >= 1:
            raise ValueError("throttle_speed_fraction must be between 0 and 1")
        if self.throttle_vibration_mms >= self.emergency_vibration_mms:
            raise ValueError("throttle vibration limit must be below emergency limit")
        if self.throttle_bearing_temp_c >= self.emergency_bearing_temp_c:
            raise ValueError("throttle temperature limit must be below emergency limit")
        if self.throttle_pressure_bar <= self.emergency_pressure_bar:
            raise ValueError("throttle pressure limit must exceed emergency limit")


@dataclass(frozen=True)
class InterlockDecision:
    action: InterlockAction
    reason: str
    validated_reading: Mapping[str, Any] | None
    speed_limit_rpm: float | None = None


class SafetyInterlockGate:
    """Deterministic, rule-based authorization gate for automatic PLC writes."""

    def __init__(self, thresholds: SafetyThresholds | None = None) -> None:
        self.thresholds = thresholds or SafetyThresholds()

    def evaluate(
        self,
        reading: Mapping[str, Any],
        *,
        llm_recommendation: str | None = None,
    ) -> InterlockDecision:
        """Return a decision from sensor physics only; LLM text cannot trigger it."""
        del llm_recommendation  # explicitly not an input to safety authorization
        if reading.get("validation_status", "OK") != "OK":
            return InterlockDecision(
                InterlockAction.NONE,
                "Telemetry status is not validated as good",
                None,
            )
        result = validate_reading(dict(reading))
        if not result.is_valid or result.cleaned_reading is None:
            return InterlockDecision(
                InterlockAction.NONE,
                f"Telemetry failed sensor validation: {result.message}",
                None,
            )
        clean = result.cleaned_reading
        vibration = max(clean["vibration_h_mms"], clean["vibration_v_mms"])
        temperature = clean["temp_bearing_degC"]
        pressure = clean["oil_pressure_bar"]
        t = self.thresholds

        if (
            vibration >= t.emergency_vibration_mms
            or temperature >= t.emergency_bearing_temp_c
            or pressure <= t.emergency_pressure_bar
        ):
            return InterlockDecision(
                InterlockAction.EMERGENCY_STOP,
                "Validated critical physical threshold exceeded: "
                f"vibration={vibration:g} mm/s, bearing_temperature={temperature:g} C, "
                f"pressure={pressure:g} bar",
                clean,
            )

        throttle_reasons = []
        if vibration >= t.throttle_vibration_mms:
            throttle_reasons.append(f"vibration={vibration:g} mm/s")
        if temperature >= t.throttle_bearing_temp_c:
            throttle_reasons.append(f"bearing_temperature={temperature:g} C")
        if pressure <= t.throttle_pressure_bar:
            throttle_reasons.append(f"pressure={pressure:g} bar")
        if throttle_reasons:
            return InterlockDecision(
                InterlockAction.THROTTLE,
                "Validated throttle threshold reached: " + ", ".join(throttle_reasons),
                clean,
                speed_limit_rpm=clean["shaft_rpm"] * t.throttle_speed_fraction,
            )
        return InterlockDecision(InterlockAction.NONE, "All validated readings within limits", clean)


class OPCUAWriteBackClient:
    """Short-lived async OPC-UA session for gated emergency-stop/speed writes."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        emergency_node_template: str | None = None,
        speed_node_template: str | None = None,
        username: str | None = None,
        password: str | None = None,
        security_string: str | None = None,
        gate: SafetyInterlockGate | None = None,
    ) -> None:
        self.endpoint = endpoint or os.getenv("OPCUA_ENDPOINT_URL", "opc.tcp://localhost:4840")
        self.emergency_node_template = emergency_node_template or os.getenv(
            "OPCUA_EMERGENCY_STOP_NODE", "ns=2;s=Assets.{asset_id}.Emergency_Stop"
        )
        self.speed_node_template = speed_node_template or os.getenv(
            "OPCUA_SPEED_LIMIT_NODE", "ns=2;s=Assets.{asset_id}.Speed_Limit_RPM"
        )
        self.username = username if username is not None else os.getenv("OPCUA_USER")
        self.password = password if password is not None else os.getenv("OPCUA_PASSWORD")
        self.security_string = security_string or os.getenv("OPCUA_SECURITY_STRING")
        self.gate = gate or SafetyInterlockGate()

    def _asset_node_id(self, template: str, asset_id: str) -> str:
        if not _SAFE_ASSET_ID.fullmatch(asset_id):
            raise ValueError("asset_id contains characters unsafe for an OPC-UA node id")
        try:
            return template.format(asset_id=asset_id)
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid OPC-UA node id template") from exc

    async def execute_if_required(
        self,
        asset_id: str,
        reading: Mapping[str, Any],
        *,
        llm_recommendation: str | None = None,
    ) -> InterlockDecision:
        """Evaluate first; connect/write only for an authorized deterministic action."""
        with tracer.start_as_current_span("opcua.safety_interlock.evaluate") as span:
            span.set_attribute("safety_gate.llm_authorizes_write", False)
            decision = self.gate.evaluate(reading, llm_recommendation=llm_recommendation)
            span.set_attribute("safety_gate.action", decision.action.value)
            span.set_attribute("safety_gate.reason", decision.reason)
        if decision.action is InterlockAction.NONE:
            LOGGER.info("Safety gate blocked PLC action for %s: %s", asset_id, decision.reason)
            return decision
        if not _SAFE_ASSET_ID.fullmatch(asset_id):
            raise ValueError("asset_id contains characters unsafe for an OPC-UA node id")

        client = Client(self.endpoint)
        if self.security_string:
            with tracer.start_as_current_span("opcua.configure_security"):
                await client.set_security_string(self.security_string)
        elif self.username and self.password:
            client.set_user(self.username)
            client.set_password(self.password)
        else:
            LOGGER.warning("OPC-UA is connecting without configured message security or credentials")

        try:
            with tracer.start_as_current_span("opcua.client_session") as span:
                span.set_attribute("opcua.endpoint", self.endpoint)
                async with client:
                    if decision.action is InterlockAction.EMERGENCY_STOP:
                        node_id = self._asset_node_id(self.emergency_node_template, asset_id)
                        with tracer.start_as_current_span("opcua.write_emergency_stop") as write_span:
                            write_span.set_attribute("opcua.asset_id", asset_id)
                            write_span.set_attribute("opcua.node_id", node_id)
                            await client.get_node(node_id).write_value(True, VariantType.Boolean)
                    elif decision.action is InterlockAction.THROTTLE:
                        node_id = self._asset_node_id(self.speed_node_template, asset_id)
                        assert decision.speed_limit_rpm is not None
                        with tracer.start_as_current_span("opcua.write_speed_limit") as write_span:
                            write_span.set_attribute("opcua.asset_id", asset_id)
                            write_span.set_attribute("opcua.node_id", node_id)
                            write_span.set_attribute("opcua.speed_limit_rpm", decision.speed_limit_rpm)
                            await client.get_node(node_id).write_value(decision.speed_limit_rpm, VariantType.Double)
        except Exception as exc:
            with tracer.start_as_current_span("opcua.write_failure") as span:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        LOGGER.critical("PLC %s command issued for %s: %s", decision.action.value, asset_id, decision.reason)
        return decision


async def apply_safety_interlock(
    asset_id: str,
    reading: Mapping[str, Any],
    *,
    llm_recommendation: str | None = None,
    client: OPCUAWriteBackClient | None = None,
) -> InterlockDecision:
    """Convenience API for backend callers; all actions still pass through the gate."""
    writer = client or OPCUAWriteBackClient()
    return await writer.execute_if_required(
        asset_id,
        reading,
        llm_recommendation=llm_recommendation,
    )

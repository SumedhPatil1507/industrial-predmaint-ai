"""Compatibility facade for the safety-gated OPC-UA writer.

All write APIs require a validated sensor reading. A caller-provided reason or
LLM recommendation is retained for logging only and never authorizes a write.
"""
from __future__ import annotations

from typing import Any, Mapping

from backend.opcua_client import (
    InterlockDecision,
    OPCUAWriteBackClient,
    SafetyInterlockGate,
    SafetyThresholds,
)


class OPCUAWriteBack:
    """Legacy name backed exclusively by ``SafetyInterlockGate``."""

    def __init__(
        self,
        endpoint: str | None = None,
        node_id: str | None = None,
        auth: Mapping[str, str | None] | None = None,
        *,
        gate: SafetyInterlockGate | None = None,
    ) -> None:
        auth = auth or {}
        self.client = OPCUAWriteBackClient(
            endpoint,
            emergency_node_template=node_id,
            username=auth.get("username"),
            password=auth.get("password"),
            gate=gate,
        )

    async def trigger_emergency_stop(
        self,
        asset_id: str,
        reading: Mapping[str, Any],
        reason: str | None = None,
        *,
        llm_recommendation: str | None = None,
    ) -> InterlockDecision:
        """Evaluate validated sensor physics; ``reason`` is not authorization."""
        del reason
        return await self.client.execute_if_required(
            asset_id,
            reading,
            llm_recommendation=llm_recommendation,
        )


ot_writeback = OPCUAWriteBack()


async def trigger_emergency_stop(
    asset_id: str,
    reading: Mapping[str, Any],
    reason: str | None = None,
    *,
    llm_recommendation: str | None = None,
) -> InterlockDecision:
    """Module-level compatibility API that cannot write without sensor data."""
    return await ot_writeback.trigger_emergency_stop(
        asset_id,
        reading,
        reason,
        llm_recommendation=llm_recommendation,
    )


__all__ = [
    "InterlockDecision",
    "OPCUAWriteBack",
    "SafetyInterlockGate",
    "SafetyThresholds",
    "ot_writeback",
    "trigger_emergency_stop",
]

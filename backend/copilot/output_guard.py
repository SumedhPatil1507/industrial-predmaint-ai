"""Pydantic schema and source-provenance checks for maintenance work orders."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict


class WorkOrderOutput(BaseModel):
    """The structured contract emitted by the Copilot work-order node."""

    model_config = ConfigDict(extra="allow")
    work_order_id: str
    asset_tag: str
    machine_type: str
    priority: str
    matched_sop_id: str | None
    safety_protocol: str
    recommended_actions: list[str]
    required_parts: list[str]
    manual_citations: list[str]


@lru_cache(maxsize=1)
def _guardrails_schema_guard():
    """Build Guardrails when its optional Pydantic adapter is available."""
    from guardrails import Guard

    # Guardrails' public API has varied across releases, and some versions
    # expose ``Guard`` as a Pydantic model without the from_pydantic adapter.
    # The Pydantic validation below remains authoritative in that case.
    from_pydantic = getattr(Guard, "from_pydantic", None)
    if not callable(from_pydantic):
        return None
    return from_pydantic(output_class=WorkOrderOutput)


def validate_work_order(
    work_order: Mapping[str, Any],
    sources: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate structure, citations, exact-source LOTO/actions, and BOM allowlist.

    The internal manual corpus is the approved procedure and part catalog. This
    function fails closed instead of trying to repair safety-critical text.
    Guardrails is an optional additional schema check; Pydantic and the explicit
    source checks always run, including when its adapter is unavailable.
    """
    payload = dict(work_order)
    # Guardrails is an optional schema check. The same Pydantic schema and all
    # explicit citation, safety text, action, and BOM checks remain mandatory.
    # No LLM repair is permitted.
    try:
        try:
            guard = _guardrails_schema_guard()
        except ImportError:
            guard = None
        if guard is not None:
            guarded = guard.parse(json.dumps(payload))
            if getattr(guarded, "validation_passed", True) is False:
                raise ValueError("Guardrails reported a schema validation failure")
        payload = WorkOrderOutput.model_validate(payload).model_dump()
    except Exception as exc:
        raise ValueError("Work order schema validation failed") from exc

    source_by_id = {str(source.get("vector_chunk_id")): source for source in sources}
    citations = payload.get("manual_citations", [])
    citation_ids = citations
    if len(citation_ids) != len(set(citation_ids)):
        raise ValueError("manual citations contain duplicate vector chunk IDs")
    if any(not chunk_id or chunk_id not in source_by_id for chunk_id in citation_ids):
        raise ValueError("manual citation does not resolve to a retrieved vector chunk")
    if len(citations) != len(sources):
        raise ValueError("every retrieved manual excerpt must carry a vector chunk citation")

    if payload.get("matched_sop_id"):
        selected = next(
            (source for source in sources if source.get("manual_id") == payload["matched_sop_id"]),
            None,
        )
        if selected is None:
            raise ValueError("matched SOP is not present in the retrieved source set")
        if selected.get("vector_chunk_id") not in citation_ids:
            raise ValueError("matched SOP is missing its vector chunk citation")
        # Exact source equality prohibits invented or omitted LOTO requirements.
        if payload.get("safety_protocol") != selected.get("safety_protocol"):
            raise ValueError("LOTO/safety protocol must exactly match the cited OEM source")
        if payload.get("recommended_actions") != selected.get("sop_steps"):
            raise ValueError("recommended actions must exactly match the cited OEM procedure")
        catalog = set(selected.get("required_parts", []))
        parts = payload.get("required_parts", [])
        if any(not isinstance(part, str) or not part.strip() or part not in catalog for part in parts):
            raise ValueError("BOM contains a part not listed in the cited manual catalog")
    else:
        if payload.get("safety_protocol") or payload.get("recommended_actions") or payload.get("required_parts"):
            raise ValueError("uncited safety procedures and BOM parts are not permitted")
        if citations:
            raise ValueError("manual citations require a matched SOP")
    return payload

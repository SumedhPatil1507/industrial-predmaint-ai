"""Fail-closed Guardrails and source-provenance checks for maintenance work orders."""
from __future__ import annotations

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


def validate_work_order(
    work_order: Mapping[str, Any],
    sources: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate structure, citations, exact-source LOTO/actions, and BOM allowlist.

    The internal manual corpus is the approved procedure and part catalog. This
    function fails closed instead of trying to repair safety-critical text.
    """
    payload = dict(work_order)
    # Validate directly with Pydantic. Guardrails' ``Guard.from_pydantic`` is
    # not available in all supported Guardrails releases and was raising during
    # module import, preventing Streamlit from starting at all.
    try:
        payload = WorkOrderOutput.model_validate(payload).model_dump()
    except Exception as exc:
        raise ValueError("Invalid maintenance work order structure") from exc

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

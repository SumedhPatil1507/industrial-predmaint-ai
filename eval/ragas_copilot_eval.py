"""Run a 50-scenario RAGAS audit against PredMaint's manual retriever.

Usage:
    python eval/ragas_copilot_eval.py --output eval/reports/ragas-audit.json

RAGAS uses an LLM judge and embeddings; configure OPENAI_API_KEY before running.
The evaluation dataset is generated deterministically from the bundled OEM manuals.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.copilot.manuals_data import get_all_manuals
from backend.copilot.rag_engine import get_rag_indexer


def build_scenarios(count: int = 50) -> list[dict[str, Any]]:
    if count != 50:
        raise ValueError("The production audit suite is fixed at exactly 50 scenarios")
    manuals = get_all_manuals()
    if not manuals:
        raise RuntimeError("No OEM manuals are available for evaluation")
    categories = ("symptoms", "failure_mode", "safety", "parts", "tolerances")
    scenarios = []
    for index in range(count):
        manual = manuals[index % len(manuals)]
        category = categories[(index // len(manuals)) % len(categories)]
        if category == "symptoms":
            question = f"For {manual['machine_type']}, what fault is indicated by: {manual['symptoms']}?"
            reference = f"{manual['title']}. {manual['root_cause_analysis']}"
        elif category == "failure_mode":
            failure = manual["failure_modes"][index % len(manual["failure_modes"])]
            question = f"How should a technician diagnose and remediate {failure} on a {manual['machine_type']}?"
            reference = " ".join(manual["sop_steps"])
        elif category == "safety":
            question = f"What safety and LOTO controls are required for {manual['title']}?"
            reference = manual["safety_protocol"]
        elif category == "parts":
            question = f"Which OEM-listed BOM parts are required for {manual['title']}?"
            reference = "; ".join(manual["required_parts"])
        else:
            question = f"What clearances and operating tolerances apply to {manual['title']}?"
            reference = manual["clearance_tolerances"]
        scenarios.append(
            {
                "scenario_id": f"IND-{index + 1:03d}",
                "manual_id": manual["manual_id"],
                "machine_type": manual["machine_type"],
                "category": category,
                "user_input": question,
                "reference": reference,
                "source_manual": manual,
            }
        )
    return scenarios


def format_context(document: dict[str, Any]) -> str:
    fields = (
        ("Manual", "manual_id"),
        ("Chunk", "vector_chunk_id"),
        ("Title", "title"),
        ("Subsystem", "subsystem"),
        ("Snippet", "snippet"),
        ("Safety protocol", "safety_protocol"),
        ("SOP steps", "sop_steps"),
        ("BOM", "required_parts"),
        ("Tolerances", "tolerances"),
        ("Root cause", "root_cause_analysis"),
    )
    return "\n".join(f"{label}: {document.get(key, '')}" for label, key in fields)


def response_for(scenario: dict[str, Any], documents: list[dict[str, Any]]) -> str:
    if not documents:
        return "No matching OEM manual was retrieved; escalate for specialist review."
    doc = documents[0]
    category = scenario["category"]
    if category == "symptoms":
        return f"{doc.get('title', '')}. {doc.get('root_cause_analysis', '')}"
    if category == "failure_mode":
        return " ".join(doc.get("sop_steps", []))
    if category == "safety":
        return str(doc.get("safety_protocol", ""))
    if category == "parts":
        return "; ".join(doc.get("required_parts", []))
    return str(doc.get("tolerances", ""))


async def run_evaluation(output_path: Path, model_name: str = "gpt-4o-mini") -> dict[str, Any]:
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install ragas and its OpenAI integration dependencies before running the audit"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for RAGAS judge and embedding calls")
    client = AsyncOpenAI(api_key=api_key)
    evaluator_llm = llm_factory(model_name, client=client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)
    metrics = {
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings),
    }
    indexer = get_rag_indexer()
    semaphore = asyncio.Semaphore(4)
    rows: list[dict[str, Any]] = []

    async def score_metric(metric_name: str, scenario: dict[str, Any], answer: str, contexts: list[str]):
        metric = metrics[metric_name]
        args = {
            "user_input": scenario["user_input"],
            "response": answer,
            "retrieved_contexts": contexts,
        }
        if metric_name in ("context_precision", "context_recall"):
            args["reference"] = scenario["reference"]
        async with semaphore:
            result = await metric.ascore(**args)
        return float(result.value)

    async def evaluate_one(scenario: dict[str, Any]) -> dict[str, Any]:
        retrieval = indexer.search_manuals(
            machine_type=scenario["machine_type"],
            query=scenario["user_input"],
            top_k=3,
        )
        docs = retrieval["retrieved_docs"]
        contexts = [format_context(doc) for doc in docs]
        answer = response_for(scenario, docs)
        scores: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for metric_name in metrics:
            try:
                scores[metric_name] = await score_metric(metric_name, scenario, answer, contexts)
            except Exception as exc:
                scores[metric_name] = None
                errors[metric_name] = str(exc)
        return {
            "scenario_id": scenario["scenario_id"],
            "manual_id": scenario["manual_id"],
            "category": scenario["category"],
            "user_input": scenario["user_input"],
            "reference": scenario["reference"],
            "response": answer,
            "retrieved_context_ids": [doc.get("vector_chunk_id") for doc in docs],
            "retrieved_contexts": contexts,
            "scores": scores,
            "errors": errors,
        }

    rows = await asyncio.gather(*(evaluate_one(scenario) for scenario in build_scenarios()))

    metric_names = tuple(metrics)
    summary = {
        name: {
            "mean": mean(values) if values else None,
            "scored_scenarios": len(values),
            "failed_scenarios": len(rows) - len(values),
        }
        for name in metric_names
        for values in [[row["scores"][name] for row in rows if row["scores"][name] is not None]]
    }
    report = {
        "audit_type": "predmaint_ragas_copilot_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "scenario_count": len(rows),
        "metrics": summary,
        "scenarios": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    await client.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "reports" / "ragas_copilot_audit.json",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    report = asyncio.run(run_evaluation(args.output, args.model))
    print(json.dumps({"report": str(args.output), "scenario_count": report["scenario_count"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()

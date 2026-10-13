"""OpenTelemetry setup shared by backend subsystems."""
from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_tracing() -> None:
    """Install an SDK provider once, using OTLP when configured by deployment."""
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return
    provider = TracerProvider(
        resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "predmaint-ai")})
    )
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        except ImportError:
            # Keep application startup available if the optional exporter package is
            # omitted; hosts can still install their own SDK provider/exporter.
            pass
    trace.set_tracer_provider(provider)


configure_tracing()


def get_tracer(name: str):
    return trace.get_tracer(name)

"""OpenTelemetry distributed tracing setup."""

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


def setup_tracer():
    """Initializes OpenTelemetry tracer provider if endpoint is configured."""
    settings = get_settings()
    if not settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        logger.debug("OpenTelemetry endpoint not set; tracing disabled.")
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        resource = Resource.create({"service.name": "enterprise-hybrid-rag"})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(
            OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
        )
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        logger.info(f"OpenTelemetry tracing active: {settings.OTEL_EXPORTER_OTLP_ENDPOINT}")
        return trace.get_tracer("enterprise-hybrid-rag")
    except Exception as e:
        logger.warning(f"Could not initialize OpenTelemetry: {e}")
        return None

"""Request logging and execution timer middleware."""

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.core.logger import get_logger

logger = get_logger("netsentry.api.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming HTTP requests, assigns tracing ID, and adds response processing latency header."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        response = await call_next(request)

        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

        client_host = request.client.host if request.client else "unknown"
        logger.info(
            "%s %s - %d (%s) - %.2fms - ID: %s",
            request.method,
            request.url.path,
            response.status_code,
            client_host,
            process_time_ms,
            request_id[:8],
        )

        return response

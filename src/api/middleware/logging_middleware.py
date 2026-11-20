"""High-performance ASGI request logging and latency measurement middleware."""

import time
import uuid
from typing import Callable
from starlette.types import ASGIApp, Receive, Scope, Send

from src.core.logger import get_logger

logger = get_logger("netsentry.api.access")


class RequestLoggingMiddleware:
    """Pure ASGI middleware logging HTTP requests, attaching tracing ID and latency headers."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        status_code = [200]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code[0] = message.get("status", 200)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                headers.append((b"x-process-time-ms", f"{elapsed_ms:.2f}".encode()))
                message["headers"] = headers

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            client = scope.get("client")
            client_ip = client[0] if client else "unknown"
            logger.info(
                "%s %s - %d (%s) - %.2fms - ID: %s",
                scope.get("method", "GET"),
                scope.get("path", "/"),
                status_code[0],
                client_ip,
                elapsed_ms,
                request_id[:8],
            )

"""High-performance ASGI sliding window rate limiting middleware."""

import json
import time
from collections import defaultdict

from starlette.types import ASGIApp, Receive, Scope, Send


class InMemoryRateLimiterMiddleware:
    """Pure ASGI middleware enforcing client IP rate limits without thread blocking."""

    def __init__(self, app: ASGIApp, max_requests_per_minute: int = 120):
        self.app = app
        self.max_requests = max_requests_per_minute
        self.client_records: dict[str, list[float]] = defaultdict(list)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in ("/api/v1/health", "/docs", "/openapi.json", "/redoc"):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_ip = client[0] if client else "127.0.0.1"
        now = time.time()
        window_start = now - 60.0

        timestamps = self.client_records[client_ip]
        self.client_records[client_ip] = [ts for ts in timestamps if ts > window_start]

        if len(self.client_records[client_ip]) >= self.max_requests:
            body = json.dumps({
                "error": "RateLimitExceeded",
                "message": f"Rate limit of {self.max_requests} requests per minute exceeded.",
                "retry_after_seconds": 60,
            }).encode("utf-8")

            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"retry-after", b"60"),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            })
            return

        self.client_records[client_ip].append(now)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                remaining = max(0, self.max_requests - len(self.client_records[client_ip]))
                headers = list(message.get("headers", []))
                headers.append((b"x-ratelimit-limit", str(self.max_requests).encode()))
                headers.append((b"x-ratelimit-remaining", str(remaining).encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)

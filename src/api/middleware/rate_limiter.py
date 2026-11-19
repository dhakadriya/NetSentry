"""In-memory sliding window rate limiting middleware."""

import time
from collections import defaultdict
from typing import Dict, List
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class InMemoryRateLimiterMiddleware(BaseHTTPMiddleware):
    """Enforces per-client IP rate limits across API endpoints."""

    def __init__(self, app, max_requests_per_minute: int = 120):
        super().__init__(app)
        self.max_requests = max_requests_per_minute
        self.client_records: Dict[str, List[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Exempt health check and documentation endpoints from rate limiting
        if request.url.path in ("/api/v1/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        window_start = now - 60.0

        # Purge timestamps older than 60 seconds
        timestamps = self.client_records[client_ip]
        self.client_records[client_ip] = [ts for ts in timestamps if ts > window_start]

        if len(self.client_records[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "message": f"Rate limit of {self.max_requests} requests per minute exceeded.",
                    "retry_after_seconds": 60,
                },
                headers={"Retry-After": "60"},
            )

        self.client_records[client_ip].append(now)
        response = await call_next(request)

        remaining = max(0, self.max_requests - len(self.client_records[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

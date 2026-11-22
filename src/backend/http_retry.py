'''Retry temporary HTTP failures for operations safe to repeat.'''

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

import httpx

Result = TypeVar("Result")


# Run an idempotent operation with a bounded backoff for transport and server errors.
async def retry_http(
    operation: Callable[[], Awaitable[Result]], delays: Sequence[float]
) -> Result:
    pending = iter(delays)
    while True:
        try:
            return await operation()
        except httpx.HTTPError as error:
            if isinstance(error, httpx.HTTPStatusError):
                code = error.response.status_code
                if code not in {408, 429} and code < 500:
                    raise
            delay = next(pending, None)
            if delay is None:
                raise
            await asyncio.sleep(delay)

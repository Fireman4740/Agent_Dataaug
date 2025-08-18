from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
import time


class AsyncRateLimiter:
    def __init__(self, calls_per_minute: Optional[int]):
        self.enabled = bool(calls_per_minute and calls_per_minute > 0)
        if self.enabled:
            assert calls_per_minute is not None
            self.interval = 60.0 / float(calls_per_minute)
        else:
            self.interval = 0.0
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def acquire(self):
        if not self.enabled:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = max(0.0, self._next_time - now)
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_time = now + self.interval


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        calls_per_minute: Optional[int] = None,
    ):
        self.api_key = api_key
        self.semaphore = semaphore or asyncio.Semaphore(5)
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.rate_limiter = AsyncRateLimiter(calls_per_minute)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((aiohttp.ClientResponseError, aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        t0 = time.perf_counter()
        await self.rate_limiter.acquire()
        logger.info(
            "[LLM] chat START model={} temp={} messages={} max_tokens={}",
            model,
            temperature,
            len(messages),
            max_tokens,
        )
        try:
            async with self.semaphore:
                timeout = aiohttp.ClientTimeout(total=300, connect=30, sock_connect=30, sock_read=300)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.base_url, headers=headers, json=payload) as resp:
                        if resp.status == 429:
                            # Respect Retry-After if present
                            retry_after = resp.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    seconds = float(retry_after)
                                except Exception:
                                    seconds = 5.0
                            else:
                                seconds = 5.0
                            logger.warning("[LLM] 429 Too Many Requests, sleeping {:.1f}s", seconds)
                            await asyncio.sleep(seconds)
                            resp.raise_for_status()  # trigger tenacity retry
                        resp.raise_for_status()
                        data = await resp.json()
                        # Optional usage stats
                        usage = data.get("usage") or {}
                        if usage:
                            logger.info(
                                "[LLM] usage prompt_tokens={} completion_tokens={} total_tokens={}",
                                usage.get("prompt_tokens"),
                                usage.get("completion_tokens"),
                                usage.get("total_tokens"),
                            )
                        return data["choices"][0]["message"]["content"]
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("[LLM] chat END   model={} duration_ms={:.1f}", model, dt_ms)

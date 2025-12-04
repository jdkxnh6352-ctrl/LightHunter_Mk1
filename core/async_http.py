# -*- coding: utf-8 -*-
"""
core/async_http.py

LightHunter Async HTTP 内核 (Mk2)

职责：
    - 提供基于 aiohttp 的异步 HTTP 客户端；
    - 支持全局并发控制（max_inflight）与超时控制；
    - 返回一个轻量级的 AsyncHTTPResponse 对象，封装 status/json/text 等方法。

注意：
    - 这里只负责“发请求 + 收响应体”，不关心业务逻辑（站点策略、重试、限流等）；
    - 上层 RequestEngine 负责：站点识别、限速、重试策略、UA 池、代理等；
    - 需要依赖 aiohttp：
        pip install aiohttp
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp

from core.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class AsyncHTTPConfig:
    """Async HTTP 全局配置"""

    timeout: float = 10.0
    max_inflight: int = 128
    verify_ssl: bool = True


class AsyncHTTPError(Exception):
    """Async HTTP 请求错误"""

    def __init__(self, status: int, url: str, message: str = "") -> None:
        self.status = status
        self.url = url
        self.message = message or f"HTTP {status} for {url}"
        super().__init__(self.message)


@dataclass
class AsyncHTTPResponse:
    """
    轻量封装 aiohttp 的 Response 内容。

    属性：
        url: 最终 URL（含重定向后）
        status: HTTP 状态码
        headers: 响应头（普通 dict）
        content: 原始字节内容
    """

    url: str
    status: int
    headers: Dict[str, str]
    content: bytes
    reason: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 400

    def text(self, encoding: Optional[str] = None, errors: str = "replace") -> str:
        if encoding is None:
            # 简单从 headers 里猜 charset
            ctype = self.headers.get("Content-Type", "")
            if "charset=" in ctype:
                encoding = ctype.split("charset=")[-1].split(";")[0].strip()
            else:
                encoding = "utf-8"
        try:
            return self.content.decode(encoding, errors=errors)
        except Exception:
            # 兜底
            return self.content.decode("utf-8", errors=errors)

    def json(self, encoding: Optional[str] = None) -> Any:
        text = self.text(encoding=encoding)
        return json.loads(text)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise AsyncHTTPError(self.status, self.url, self.reason)


class AsyncHTTPClient:
    """
    基于 aiohttp 的异步 HTTP 客户端。

    特点：
        - 懒加载 ClientSession，按需创建；
        - 通过 asyncio.Semaphore 控制并发请求数（max_inflight）；
        - 每次 request 返回 AsyncHTTPResponse（已读完响应体，便于上层使用）。
    """

    def __init__(self, config: Optional[AsyncHTTPConfig] = None) -> None:
        self.cfg = config or AsyncHTTPConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._sem: Optional[asyncio.Semaphore] = None
        if self.cfg.max_inflight and self.cfg.max_inflight > 0:
            self._sem = asyncio.Semaphore(self.cfg.max_inflight)

        self._closed = False
        self.log = get_logger(self.__class__.__name__)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise RuntimeError("AsyncHTTPClient 已关闭，不能再发请求")
        if self._session is not None and not self._session.closed:
            return self._session

        timeout = aiohttp.ClientTimeout(total=self.cfg.timeout)
        connector = aiohttp.TCPConnector(ssl=self.cfg.verify_ssl)
        self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        self.log.info(
            "AsyncHTTPClient: 创建 aiohttp 会话 timeout=%.1f max_inflight=%s verify_ssl=%s",
            self.cfg.timeout,
            self.cfg.max_inflight,
            self.cfg.verify_ssl,
        )
        return self._session

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncHTTPResponse:
        """
        发送异步 HTTP 请求。

        Args:
            method: "GET"/"POST"/...
            url: 完整 URL
            params/data/json/headers/timeout: 对应 aiohttp 的参数
            kwargs: 透传给 aiohttp.ClientSession.request

        Returns:
            AsyncHTTPResponse
        """
        session = await self._ensure_session()
        method = method.upper()

        # 单次请求可覆盖 total timeout
        t = aiohttp.ClientTimeout(total=timeout or self.cfg.timeout)

        async def _do_request() -> AsyncHTTPResponse:
            async with session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json,
                headers=headers,
                timeout=t,
                **kwargs,
            ) as resp:
                content = await resp.read()
                return AsyncHTTPResponse(
                    url=str(resp.url),
                    status=resp.status,
                    headers={k: v for k, v in resp.headers.items()},
                    content=content,
                    reason=resp.reason or "",
                )

        if self._sem is None:
            return await _do_request()

        async with self._sem:
            return await _do_request()

    async def get(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncHTTPResponse:
        return await self.request(
            "GET", url, params=params, headers=headers, timeout=timeout, **kwargs
        )

    async def post(
        self,
        url: str,
        *,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncHTTPResponse:
        return await self.request(
            "POST", url, data=data, json=json, headers=headers, timeout=timeout, **kwargs
        )

    async def aclose(self) -> None:
        """关闭底层 aiohttp.ClientSession。"""
        self._closed = True
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self.log.info("AsyncHTTPClient: aiohttp 会话已关闭")

    async def __aenter__(self) -> "AsyncHTTPClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

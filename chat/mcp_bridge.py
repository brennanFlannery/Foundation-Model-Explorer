"""In-process MCP bridge — runs FastMCP server in a background thread."""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List


class MCPBridge:
    """Synchronous facade around an in-process FastMCP server.

    The server runs in a dedicated event loop on a background daemon thread.
    list_tools() and call_tool() submit coroutines to that loop and block
    until results are available.  Public API is identical to the previous
    subprocess-based bridge so call sites need no changes.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="mcp-bridge-loop", daemon=True
        )
        self._session: Any = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout_s: float = 10.0) -> None:
        """Start background event loop and wait for session to be ready."""
        self._thread.start()
        if not self._ready.wait(timeout=timeout_s):
            raise RuntimeError("In-process MCP server failed to start within timeout")

    def stop(self) -> None:
        """Signal the session to close and wait for the thread to exit."""
        if self._stop_event is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop_event.set)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run_session())

    async def _run_session(self) -> None:
        try:
            from mcp.shared.memory import (  # type: ignore
                create_connected_server_and_client_session,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MCP SDK is not installed. Install `mcp` to enable local tools."
            ) from exc

        from .mcp_server import mcp as _mcp_app  # import here to avoid circular at module level

        self._stop_event = asyncio.Event()
        async with create_connected_server_and_client_session(_mcp_app) as client_session:
            self._session = client_session
            self._ready.set()
            await self._stop_event.wait()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of tool descriptors (name, description, input_schema)."""
        self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(
            self._list_tools_async(), self._loop
        )
        return future.result(timeout=10.0)

    def call_tool(
        self, name: str, arguments: Dict[str, Any], timeout_s: float = 30.0
    ) -> Dict[str, Any]:
        """Call a named tool and return the parsed JSON result."""
        self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(name, arguments), self._loop
        )
        return future.result(timeout=timeout_s)

    def _ensure_started(self) -> None:
        """Start the bridge on first use if not already running."""
        if not self._ready.is_set():
            self.start()

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------

    async def _list_tools_async(self) -> List[Dict[str, Any]]:
        response = await self._session.list_tools()
        tools: List[Dict[str, Any]] = []
        for tool in getattr(response, "tools", []) or []:
            tools.append(
                {
                    "name": getattr(tool, "name", ""),
                    "description": getattr(tool, "description", ""),
                    "input_schema": getattr(tool, "inputSchema", None)
                    or getattr(tool, "input_schema", None)
                    or {"type": "object", "properties": {}},
                }
            )
        return tools

    async def _call_tool_async(
        self, name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        response = await self._session.call_tool(name, arguments)
        raw_content = getattr(response, "content", None)
        if isinstance(raw_content, list) and raw_content:
            text = getattr(raw_content[0], "text", None)
            if text:
                try:
                    return json.loads(text)
                except Exception:
                    return {"result": text}
        if isinstance(raw_content, dict):
            return raw_content
        return {"result": str(response)}

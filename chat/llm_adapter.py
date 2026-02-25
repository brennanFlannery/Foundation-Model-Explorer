"""LiteLLM adapter for provider-agnostic chat completions with tool calls."""
from __future__ import annotations

import json
from typing import Any, Dict, Generator, Iterable, List, Optional


class LiteLLMClient:
    """Thin wrapper around LiteLLM chat completion APIs."""

    def __init__(self, model: str, api_key: str, timeout_s: float = 60.0) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _import_litellm(self):
        try:
            from litellm import completion  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "LiteLLM is not installed. Install `litellm` to use chat integration."
            ) from exc
        return completion

    def stream_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        timeout_s: Optional[float] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield streaming deltas and any tool-call metadata in provider format."""
        completion = self._import_litellm()
        stream = completion(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=True,
            api_key=self.api_key,
            timeout=timeout_s or self.timeout_s,
        )
        for chunk in stream:
            yield chunk

    def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a non-streamed completion and return raw response object."""
        completion = self._import_litellm()
        return completion(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=False,
            api_key=self.api_key,
            timeout=timeout_s or self.timeout_s,
        )

    @staticmethod
    def normalize_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize tool calls to a provider-agnostic shape.

        Returns list entries shaped as:
        {"call_id": str, "name": str, "arguments": Dict[str, Any]}
        """
        normalized: List[Dict[str, Any]] = []

        choices = response.get("choices") or []
        if not choices:
            return normalized

        message = (choices[0] or {}).get("message") or {}
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            args_raw = function.get("arguments") or "{}"
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {}
            normalized.append(
                {
                    "call_id": call.get("id") or "",
                    "name": function.get("name") or "",
                    "arguments": args,
                }
            )
        return normalized

    @staticmethod
    def normalize_usage(response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract usage payload where present."""
        usage = response.get("usage") or {}
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    @staticmethod
    def extract_text_delta(chunk: Dict[str, Any]) -> str:
        """Extract text delta from a streaming response chunk."""
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        delta = (choices[0] or {}).get("delta") or {}
        return delta.get("content") or ""

    @staticmethod
    def extract_stream_tool_calls(chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect any streamed tool call fragments if provider emits them."""
        # MVP keeps this intentionally simple and expects tool calls from
        # non-streamed step when needed. This parser is a lightweight fallback.
        calls: List[Dict[str, Any]] = []
        for chunk in chunks:
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            if "tool_calls" in delta:
                calls.extend(delta.get("tool_calls") or [])
        return calls

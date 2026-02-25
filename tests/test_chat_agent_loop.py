from __future__ import annotations

from typing import Any, Dict, List

from chat import ChatAgentConfig, ChatAgentWorker


class FakeLLM:
    def __init__(self) -> None:
        self._responses: List[Dict[str, Any]] = []

    def queue_response(self, response: Dict[str, Any]) -> None:
        self._responses.append(response)

    def complete_with_tools(self, messages, tools, timeout_s):
        if not self._responses:
            return {"choices": [{"message": {"content": "No response queued"}}]}
        return self._responses.pop(0)

    def normalize_tool_calls(self, response):
        return response.get("tool_calls", [])

    def stream_with_tools(self, messages, tools, timeout_s):
        for token in ["Hello", " ", "world"]:
            yield {"choices": [{"delta": {"content": token}}]}

    def extract_text_delta(self, chunk):
        return chunk["choices"][0]["delta"]["content"]

    def normalize_usage(self, response):
        return {"total_tokens": 3}


class FakeMCP:
    def start(self):
        return None

    def list_tools(self):
        return [
            {
                "name": "list_data",
                "description": "List data",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

    def call_tool(self, name, arguments, timeout_s):
        return {"count_summary": {"slides": 1, "feature_files": 1}}

    def stop(self):
        return None


def test_chat_agent_runs_tool_and_streams_response():
    config = ChatAgentConfig(
        model="gpt-4o-mini",
        api_key="test",
    )
    worker = ChatAgentWorker(config)
    worker._llm = FakeLLM()  # type: ignore[attr-defined]
    worker._mcp = FakeMCP()  # type: ignore[attr-defined]

    started = []
    deltas = []
    finished = []
    errors = []

    worker.response_started.connect(lambda mid: started.append(mid))
    worker.response_delta.connect(lambda mid, delta: deltas.append(delta))
    worker.response_finished.connect(lambda mid, text, usage, latency: finished.append(text))
    worker.error_emitted.connect(lambda *args: errors.append(args))

    worker._llm.queue_response(  # type: ignore[attr-defined]
        {"tool_calls": [{"call_id": "c1", "name": "list_data", "arguments": {}}]}
    )
    worker._llm.queue_response(  # type: ignore[attr-defined]
        {"choices": [{"message": {"content": "summary"}}]}
    )

    worker.submit_user_message("Inspect", {"root_dir": "/tmp"})

    assert errors == []
    assert started
    assert "".join(deltas) == "Hello world"
    assert finished == ["Hello world"]


def test_chat_agent_emits_error_for_disallowed_tool():
    config = ChatAgentConfig(
        model="gpt-4o-mini",
        api_key="test",
    )
    worker = ChatAgentWorker(config)
    worker._llm = FakeLLM()  # type: ignore[attr-defined]
    worker._mcp = FakeMCP()  # type: ignore[attr-defined]

    errors = []
    worker.error_emitted.connect(lambda *args: errors.append(args))

    worker._llm.queue_response(  # type: ignore[attr-defined]
        {"tool_calls": [{"call_id": "c1", "name": "delete_everything", "arguments": {}}]}
    )

    worker.submit_user_message("Inspect", {"root_dir": "/tmp"})

    assert errors


def test_chat_agent_forced_slash_tool_call():
    config = ChatAgentConfig(model="gpt-4o-mini", api_key="test")
    worker = ChatAgentWorker(config)
    worker._llm = FakeLLM()  # type: ignore[attr-defined]
    worker._mcp = FakeMCP()  # type: ignore[attr-defined]

    started = []
    worker.response_started.connect(lambda mid: started.append(mid))
    worker._llm.queue_response(  # type: ignore[attr-defined]
        {"choices": [{"message": {"content": "summary"}}]}
    )

    worker.submit_user_message(
        "/data",
        {
            "root_dir": "/tmp",
            "slash_command": {
                "kind": "tool",
                "tool_name": "list_data",
                "arguments": {},
            },
        },
    )

    assert started


def test_chat_agent_slash_help_short_circuit():
    config = ChatAgentConfig(model="gpt-4o-mini", api_key="test")
    worker = ChatAgentWorker(config)
    worker._llm = FakeLLM()  # type: ignore[attr-defined]
    worker._mcp = FakeMCP()  # type: ignore[attr-defined]

    finished = []
    worker.response_finished.connect(lambda _id, text, _usage, _lat: finished.append(text))

    worker.submit_user_message(
        "/tools",
        {"root_dir": "/tmp", "slash_command": {"kind": "help", "raw": "/tools"}},
    )

    assert finished
    assert "Slash commands:" in finished[0]

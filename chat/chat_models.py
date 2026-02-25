"""Chat data models for FoundationDetector chat integration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    """Represents one chat message in session state."""

    role: str
    content: str
    message_id: str
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    """Represents a model-requested tool call."""

    call_id: str
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResultCard:
    """Tool output rendered in chat UI."""

    call_id: str
    tool_name: str
    summary: Dict[str, Any]
    raw: Dict[str, Any]


@dataclass
class ChatSessionState:
    """In-memory session state for chat interactions."""

    messages: List[ChatMessage] = field(default_factory=list)
    running: bool = False
    current_request_id: Optional[str] = None

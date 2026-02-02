"""Server-Sent Events (SSE) broadcaster for dashboard.

This module implements SSE broadcasting for real-time updates:
- Live bmad-assist output streaming
- Connection management with automatic cleanup

Public API:
    SSEBroadcaster: Main broadcaster class for managing SSE connections
    SSEMessage: Dataclass representing an SSE message
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(Enum):
    """SSE event types."""

    OUTPUT = "output"  # bmad-assist stdout/stderr
    STATUS = "status"  # General status update
    HEARTBEAT = "heartbeat"  # Keep-alive ping
    MODEL_STARTED = "model_started"  # New model invocation tab


@dataclass
class SSEMessage:
    """Represents an SSE message.

    Attributes:
        event: Event type name.
        data: Message payload (will be JSON-encoded).
        id: Optional message ID for client reconnection.
        retry: Optional retry interval in milliseconds.

    """

    event: str
    data: Any
    id: str | None = None
    retry: int | None = None

    def format(self) -> str:
        """Format message for SSE protocol.

        Returns:
            SSE-formatted string ready for transmission.

        """
        lines = []

        if self.id:
            lines.append(f"id: {self.id}")
        if self.retry:
            lines.append(f"retry: {self.retry}")

        lines.append(f"event: {self.event}")

        # Handle multi-line data
        data_str = json.dumps(self.data) if isinstance(self.data, (dict, list)) else str(self.data)

        for line in data_str.split("\n"):
            lines.append(f"data: {line}")

        lines.append("")  # Empty line terminates message
        return "\n".join(lines) + "\n"


class SSEBroadcaster:
    """Manages SSE connections and broadcasts messages.

    Async-safe broadcaster (single event loop) that maintains a set of active
    connections and broadcasts messages to all connected clients.

    Attributes:
        connection_count: Number of active connections.

    """

    def __init__(self, heartbeat_interval: float = 30.0) -> None:
        """Initialize broadcaster.

        Args:
            heartbeat_interval: Seconds between heartbeat messages.

        """
        self._queues: set[asyncio.Queue[SSEMessage | None]] = set()
        self._heartbeat_interval = heartbeat_interval
        self._message_counter = 0
        self._lock = asyncio.Lock()

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._queues)

    async def subscribe(self) -> AsyncGenerator[str, None]:
        """Subscribe to SSE stream.

        Yields:
            Formatted SSE messages as strings.

        """
        queue: asyncio.Queue[SSEMessage | None] = asyncio.Queue()

        async with self._lock:
            self._queues.add(queue)
            logger.info("SSE client connected (total: %d)", len(self._queues))

        try:
            # Send initial connection message
            yield SSEMessage(
                event=EventType.STATUS.value,
                data={"connected": True, "timestamp": time.time()},
                retry=3000,
            ).format()

            while True:
                try:
                    # Wait for message with heartbeat timeout
                    message = await asyncio.wait_for(queue.get(), timeout=self._heartbeat_interval)

                    if message is None:
                        # Shutdown signal
                        break

                    yield message.format()

                except TimeoutError:
                    # Send heartbeat
                    yield SSEMessage(
                        event=EventType.HEARTBEAT.value,
                        data={"timestamp": time.time()},
                    ).format()

        finally:
            async with self._lock:
                self._queues.discard(queue)
                logger.info("SSE client disconnected (remaining: %d)", len(self._queues))

    async def broadcast(self, event: EventType, data: Any) -> int:
        """Broadcast message to all connected clients.

        Args:
            event: Event type.
            data: Message payload.

        Returns:
            Number of clients message was sent to.

        """
        self._message_counter += 1
        message = SSEMessage(
            event=event.value,
            data=data,
            id=str(self._message_counter),
        )

        async with self._lock:
            for queue in self._queues:
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning("SSE queue full, dropping message for slow client")

            return len(self._queues)

    async def broadcast_output(
        self,
        line: str,
        provider: str | None = None,
        model_tab_id: str | None = None,
    ) -> int:
        """Broadcast bmad-assist output line.

        Args:
            line: Output line text.
            provider: Provider name (opus, gemini, etc.) or None.
            model_tab_id: Dynamic model tab ID (e.g., "opus-1", "glm-4.7-2").

        Returns:
            Number of clients message was sent to.

        """
        return await self.broadcast(
            EventType.OUTPUT,
            {
                "line": line,
                "provider": provider,
                "model_tab_id": model_tab_id,
                "timestamp": time.time(),
            },
        )

    async def broadcast_model_started(
        self,
        model: str,
        tab_id: str,
        role: str | None = None,
        provider: str | None = None,
    ) -> int:
        """Broadcast model_started event for dynamic tab creation.

        Args:
            model: Model identifier (e.g., "opus", "glm-4.7").
            tab_id: Unique tab ID (e.g., "opus-1", "glm-4.7-2").
            role: Optional role descriptor (e.g., "master", "helper").
            provider: Optional provider name.

        Returns:
            Number of clients message was sent to.

        """
        data: dict[str, Any] = {
            "model": model,
            "tab_id": tab_id,
            "timestamp": time.time(),
        }
        if role:
            data["role"] = role
        if provider:
            data["provider"] = provider

        return await self.broadcast(EventType.MODEL_STARTED, data)

    async def broadcast_event(self, event_name: str, data: dict[str, Any]) -> int:
        """Broadcast a custom event to all connected clients.

        Args:
            event_name: Custom event name (e.g., "config_reloaded").
            data: Event payload dictionary.

        Returns:
            Number of clients message was sent to.

        """
        self._message_counter += 1
        message = SSEMessage(
            event=event_name,
            data=data,
            id=str(self._message_counter),
        )

        async with self._lock:
            for queue in self._queues:
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning("SSE queue full, dropping message for slow client")

            return len(self._queues)

    async def shutdown(self) -> None:
        """Shutdown broadcaster and disconnect all clients."""
        async with self._lock:
            for queue in self._queues:
                await queue.put(None)  # Send shutdown signal

            logger.info("SSE broadcaster shutdown, disconnected %d clients", len(self._queues))
            self._queues.clear()

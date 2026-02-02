"""Cancellation context for thread-safe loop control.

Provides CancellationContext class for LoopController integration.
Allows dashboard to request cancellation and cleanup of running operations.

Safe Checkpoints (cancel honored here):
1. Before execute_phase() - state at phase boundary
2. After save_state() - state persisted to disk
3. Between stories - story boundary
4. During pause wait - in wait_for_resume()

Unsafe Points (cancel NOT honored):
1. During save_state() - would corrupt state file
2. During provider.invoke() - handled separately by provider cancel
3. During file writes - atomic writes, but incomplete operation
"""

import logging
import threading
from collections.abc import Callable

from bmad_assist.core.exceptions import CancelledError

__all__ = [
    "CancellationContext",
]

logger = logging.getLogger(__name__)


class CancellationContext:
    """Thread-safe cancellation token with cleanup hooks.

    Provides a mechanism for the main thread (dashboard) to request cancellation
    of operations running in a worker thread (orchestrator). Uses threading.Event
    for thread-safe signaling.

    Cleanup hooks are run in LIFO order (last registered = first run) to allow
    proper unwinding of nested operations (e.g., terminate child process before
    closing pipe).

    Usage:
        ctx = CancellationContext()
        ctx.register_cleanup(lambda: process.terminate())

        # In worker thread:
        if ctx.is_cancelled:
            return

        # To cancel from main thread:
        ctx.request_cancel()
        ctx.run_cleanup()  # Executes all hooks, catches exceptions

    Attributes:
        is_cancelled: Property that returns True if cancellation was requested.

    """

    def __init__(self) -> None:
        """Initialize CancellationContext with threading primitives."""
        self._cancel_event = threading.Event()
        self._cleanup_hooks: list[Callable[[], None]] = []
        self._lock = threading.Lock()  # Protects _cleanup_hooks

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested. Thread-safe.

        Returns:
            True if request_cancel() was called, False otherwise.

        """
        return self._cancel_event.is_set()

    def request_cancel(self) -> None:
        """Request cancellation. Thread-safe, idempotent.

        Can be called from any thread. Sets the cancel event which
        can be checked via is_cancelled property.
        """
        logger.info("Cancellation requested")
        self._cancel_event.set()

    def register_cleanup(self, hook: Callable[[], None]) -> None:
        """Register cleanup hook. Thread-safe.

        Hooks are stored in a list and run in LIFO order (last registered
        runs first) when run_cleanup() is called.

        Args:
            hook: Callable with no arguments to run during cleanup.

        """
        with self._lock:
            self._cleanup_hooks.append(hook)
            logger.debug("Registered cleanup hook: %s", hook)

    def run_cleanup(self) -> list[Exception]:
        """Run all cleanup hooks. Returns list of exceptions (if any).

        Hooks are run in LIFO order (last registered = first run).
        All hooks run even if some raise exceptions - exceptions are
        collected and returned.

        After running, the hooks list is cleared.

        Returns:
            List of exceptions raised by hooks (empty if all succeeded).

        """
        exceptions: list[Exception] = []

        with self._lock:
            hooks = list(reversed(self._cleanup_hooks))
            self._cleanup_hooks.clear()

        logger.info("Running %d cleanup hooks", len(hooks))

        for hook in hooks:
            try:
                hook()
            except Exception as e:
                logger.warning("Cleanup hook failed: %s", e)
                exceptions.append(e)

        return exceptions

    def check_cancelled(self) -> None:
        """Raise CancelledError if cancelled. Use at safe checkpoints.

        This method should only be called at safe checkpoints where
        raising an exception won't corrupt state.

        Raises:
            CancelledError: If cancellation was requested.

        """
        if self.is_cancelled:
            raise CancelledError("Operation cancelled by user")

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation with optional timeout.

        Useful for blocking operations that need to check for cancellation
        periodically.

        Args:
            timeout: Maximum seconds to wait, or None for no timeout.

        Returns:
            True if cancelled, False if timeout expired.

        """
        return self._cancel_event.wait(timeout)

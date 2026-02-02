"""Tests for CancellationContext.

Story: Direct Orchestrator Integration for Dashboard.
"""

import threading
import time

import pytest

from bmad_assist.core.exceptions import CancelledError
from bmad_assist.core.loop.cancellation import CancellationContext


class TestCancellationContext:
    """Tests for CancellationContext class."""

    def test_initial_state_not_cancelled(self):
        """Context starts in non-cancelled state."""
        ctx = CancellationContext()
        assert ctx.is_cancelled is False

    def test_request_cancel_sets_flag(self):
        """request_cancel() sets the cancelled flag."""
        ctx = CancellationContext()
        ctx.request_cancel()
        assert ctx.is_cancelled is True

    def test_request_cancel_is_idempotent(self):
        """Multiple request_cancel() calls don't cause issues."""
        ctx = CancellationContext()
        ctx.request_cancel()
        ctx.request_cancel()
        ctx.request_cancel()
        assert ctx.is_cancelled is True

    def test_check_cancelled_raises_when_cancelled(self):
        """check_cancelled() raises CancelledError when cancelled."""
        ctx = CancellationContext()
        ctx.request_cancel()
        with pytest.raises(CancelledError):
            ctx.check_cancelled()

    def test_check_cancelled_does_not_raise_when_not_cancelled(self):
        """check_cancelled() does not raise when not cancelled."""
        ctx = CancellationContext()
        ctx.check_cancelled()  # Should not raise

    def test_register_cleanup_adds_hooks(self):
        """register_cleanup() adds hooks to the list."""
        ctx = CancellationContext()
        called = []
        ctx.register_cleanup(lambda: called.append(1))
        ctx.register_cleanup(lambda: called.append(2))
        assert len(called) == 0  # Not called yet

    def test_run_cleanup_calls_hooks_lifo_order(self):
        """run_cleanup() calls hooks in LIFO order."""
        ctx = CancellationContext()
        called = []
        ctx.register_cleanup(lambda: called.append(1))
        ctx.register_cleanup(lambda: called.append(2))
        ctx.register_cleanup(lambda: called.append(3))

        ctx.run_cleanup()

        # LIFO: 3, 2, 1
        assert called == [3, 2, 1]

    def test_run_cleanup_clears_hooks(self):
        """run_cleanup() clears the hooks list after running."""
        ctx = CancellationContext()
        called = []
        ctx.register_cleanup(lambda: called.append(1))

        ctx.run_cleanup()
        assert called == [1]

        # Second call should not run anything
        ctx.run_cleanup()
        assert called == [1]

    def test_run_cleanup_continues_on_exception(self):
        """run_cleanup() runs all hooks even if some raise exceptions."""
        ctx = CancellationContext()
        results = []

        ctx.register_cleanup(lambda: results.append(1))
        ctx.register_cleanup(lambda: 1 / 0)  # ZeroDivisionError
        ctx.register_cleanup(lambda: results.append(3))

        exceptions = ctx.run_cleanup()

        # All hooks ran (LIFO: 3, error, 1)
        assert results == [3, 1]
        assert len(exceptions) == 1
        assert isinstance(exceptions[0], ZeroDivisionError)

    def test_wait_returns_true_when_cancelled(self):
        """wait() returns True when cancelled within timeout."""
        ctx = CancellationContext()

        def cancel_after_delay():
            time.sleep(0.1)
            ctx.request_cancel()

        thread = threading.Thread(target=cancel_after_delay)
        thread.start()

        result = ctx.wait(timeout=1.0)
        thread.join()

        assert result is True
        assert ctx.is_cancelled is True

    def test_wait_returns_false_on_timeout(self):
        """wait() returns False when timeout expires without cancel."""
        ctx = CancellationContext()
        result = ctx.wait(timeout=0.1)
        assert result is False
        assert ctx.is_cancelled is False


class TestCancellationContextThreadSafety:
    """Thread-safety tests for CancellationContext."""

    def test_concurrent_is_cancelled_checks(self):
        """Multiple threads can check is_cancelled simultaneously."""
        ctx = CancellationContext()
        results = []

        def check_cancelled():
            for _ in range(100):
                results.append(ctx.is_cancelled)

        threads = [threading.Thread(target=check_cancelled) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 500
        assert all(r is False for r in results)

    def test_cancel_visible_across_threads(self):
        """Cancel set in one thread is visible in another."""
        ctx = CancellationContext()
        visible = threading.Event()

        def wait_for_cancel():
            while not ctx.is_cancelled:
                time.sleep(0.01)
            visible.set()

        waiter = threading.Thread(target=wait_for_cancel)
        waiter.start()

        time.sleep(0.05)
        ctx.request_cancel()

        visible.wait(timeout=1.0)
        assert visible.is_set()
        waiter.join()

    def test_concurrent_cleanup_registration(self):
        """Multiple threads can register cleanup hooks."""
        ctx = CancellationContext()
        results = []
        lock = threading.Lock()

        def register_hooks():
            for i in range(10):

                def hook(n=i):
                    with lock:
                        results.append(n)

                ctx.register_cleanup(hook)

        threads = [threading.Thread(target=register_hooks) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All hooks registered
        ctx.run_cleanup()
        assert len(results) == 50  # 5 threads * 10 hooks each


class TestCancellationContextIntegration:
    """Integration tests for CancellationContext with real scenarios."""

    def test_worker_thread_checks_cancel(self):
        """Simulate worker thread that checks for cancellation."""
        ctx = CancellationContext()
        work_done = 0

        def worker():
            nonlocal work_done
            for _ in range(100):
                if ctx.is_cancelled:
                    break
                work_done += 1
                time.sleep(0.01)

        thread = threading.Thread(target=worker)
        thread.start()

        # Cancel after some work is done
        time.sleep(0.15)
        ctx.request_cancel()

        thread.join()

        # Some work was done, but not all
        assert 0 < work_done < 100

    def test_cleanup_terminates_simulated_process(self):
        """Cleanup hook can terminate a simulated process."""
        ctx = CancellationContext()
        terminated = threading.Event()

        # Simulate process that cleanup should terminate
        ctx.register_cleanup(lambda: terminated.set())

        ctx.request_cancel()
        ctx.run_cleanup()

        assert terminated.is_set()

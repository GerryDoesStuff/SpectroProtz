"""Shared timeout guard helpers for peak fitting and file indexing."""

from __future__ import annotations

import logging
import multiprocessing
import signal
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FitTimeoutError(TimeoutError):
    """Raised when a peak fit exceeds the configured timeout."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        super().__init__(f"Fit timed out after {seconds:.1f}s")


class FileTimeoutError(TimeoutError):
    """Raised when indexing a single file exceeds the configured timeout."""

    def __init__(self, path: str, seconds: float, stage: Optional[str] = None):
        self.path = path
        self.seconds = seconds
        self.stage = stage or "unknown"
        super().__init__(f"Timed out after {seconds:.1f}s during {self.stage}")


class SigAlarmTimeoutGuard:
    """Context manager enforcing a SIGALRM timeout."""

    def __init__(self, seconds: float, timeout_error_factory: Callable[[], Exception]):
        self.seconds = float(seconds or 0.0)
        self.timeout_error_factory = timeout_error_factory
        self._enabled = bool(self.seconds and self.seconds > 0 and hasattr(signal, "SIGALRM"))
        self._previous_handler = None
        self._previous_timer = None

    def __enter__(self):
        if not self._enabled:
            return self
        self._previous_handler = signal.getsignal(signal.SIGALRM)
        self._previous_timer = signal.getitimer(signal.ITIMER_REAL)

        def _handle_alarm(_signum, _frame):
            raise self.timeout_error_factory()

        signal.signal(signal.SIGALRM, _handle_alarm)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._enabled and hasattr(signal, "SIGALRM"):
            if self._previous_timer is not None:
                signal.setitimer(signal.ITIMER_REAL, *self._previous_timer)
            if self._previous_handler is not None:
                signal.signal(signal.SIGALRM, self._previous_handler)
        return False


def run_with_multiprocessing_timeout(
    seconds: float,
    operation: Callable[[], T],
    timeout_error_factory: Callable[[], Exception],
) -> T:
    if not seconds or seconds <= 0:
        return operation()
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)

    def _worker(queue):  # type: ignore[no-untyped-def]
        try:
            result = operation()
            queue.put(("result", result))
        except Exception as exc:  # pragma: no cover - depends on runtime
            queue.put(("error", exc))

    proc = ctx.Process(target=_worker, args=(result_queue,), daemon=True)
    proc.start()
    proc.join(seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise timeout_error_factory()
    if result_queue.empty():
        raise RuntimeError("Timed operation exited without returning a result.")
    status, payload = result_queue.get()
    if status == "error":
        raise payload
    return payload


def run_with_timeout(
    seconds: float,
    operation: Callable[[], T],
    timeout_error_factory: Callable[[], Exception],
    *,
    timeout_label: str,
) -> T:
    if not seconds or seconds <= 0:
        return operation()
    if hasattr(signal, "SIGALRM"):
        with SigAlarmTimeoutGuard(seconds, timeout_error_factory):
            return operation()
    logger.warning(
        "%s timeout requested but SIGALRM unavailable; falling back to multiprocessing timeout guard "
        "(runs the operation in a child process and adds startup overhead).",
        timeout_label,
    )
    return run_with_multiprocessing_timeout(seconds, operation, timeout_error_factory)


def run_with_fit_timeout(seconds: float, operation: Callable[[], T]) -> T:
    return run_with_timeout(
        seconds,
        operation,
        lambda: FitTimeoutError(seconds),
        timeout_label="Fit",
    )


def run_with_file_timeout(
    seconds: float,
    path: str,
    stage_getter: Optional[Callable[[], Optional[str]]],
    operation: Callable[[], T],
) -> T:
    def _error_factory() -> FileTimeoutError:
        stage = stage_getter() if stage_getter else None
        return FileTimeoutError(path, seconds, stage=stage)

    return run_with_timeout(
        seconds,
        operation,
        _error_factory,
        timeout_label="File",
    )

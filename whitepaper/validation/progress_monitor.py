"""Runtime monitoring utilities for long-running validation tests."""

import signal
import time
from typing import Optional


class ProgressMonitor:
    """Context manager enforcing a maximum runtime for a code block.

    Example
    -------
    >>> with ProgressMonitor(max_runtime_minutes=60):
    ...     long_running_test()
    """

    def __init__(self, max_runtime_minutes: int = 60):
        self.max_runtime = max_runtime_minutes * 60
        self._start: Optional[float] = None

    def _timeout_handler(self, signum, frame):  # pragma: no cover - signal path
        raise TimeoutError(
            f"Test exceeded {self.max_runtime/60:.0f} minute limit"
        )

    def __enter__(self):
        self._start = time.time()
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(int(self.max_runtime))
        return self

    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)
        return False

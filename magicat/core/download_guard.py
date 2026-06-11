# magicat/core/download_guard.py
"""Wall-clock watchdog for yt-dlp downloads.

socket_timeout only bounds a single socket read - under YouTube nsig
throttling the server keeps dribbling bytes, so no read ever times out.
A progress_hooks hook that raises once a wall-clock budget is exceeded is
the only reliable total-download abort (verified empirically).
"""
from __future__ import annotations

import time


class DownloadTimeout(RuntimeError):
    """A download exceeded its wall-clock budget."""


def timeout_hook(budget_s: float):
    """yt-dlp progress hook raising DownloadTimeout after budget_s."""
    deadline = time.monotonic() + budget_s

    def hook(status: dict) -> None:
        if time.monotonic() >= deadline:
            raise DownloadTimeout(
                f"download exceeded {budget_s:.0f}s budget")

    return hook

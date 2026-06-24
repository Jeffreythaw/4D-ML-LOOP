from __future__ import annotations

import logging
import os
import resource
import sys


logger = logging.getLogger("j4d.backend.memory")


def current_rss_mb() -> float:
    """Return the process max RSS in MiB using the platform's ru_maxrss units."""
    raw_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw_rss / (1024.0 * 1024.0)
    return raw_rss / 1024.0


def log_memory(label: str) -> None:
    """Emit a lightweight memory checkpoint for Render startup/prediction logs."""
    message = "memory_checkpoint label=%s rss_max_mb=%.2f pid=%s"
    args = (label, current_rss_mb(), os.getpid())
    logger.info(message, *args)
    logging.getLogger("uvicorn.error").info(message, *args)

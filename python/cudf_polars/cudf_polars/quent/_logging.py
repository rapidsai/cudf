# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

from __future__ import annotations

import collections
import threading
from typing import TYPE_CHECKING, Any

import structlog  # TODO: optional structlog.

if TYPE_CHECKING:
    from cudf_polars.quent._types import Event

QUENT_SCOPE = "QUENT"

# TODO: encapsulate this in a class?

# buffer_lock protects concurrent access to log_buffer
# This enables exactly-once read semantics from the client.
# All writers and readers must acquire the lock before accessing the buffer.
buffer_lock = threading.Lock()
# TODO: configurable maxlen?
log_buffer: collections.deque[dict[str, Any]] = collections.deque(maxlen=100_000)


def emit(event: Event) -> None:
    """Emit a Quent event."""
    logger = structlog.get_logger()
    logger.info(event.to_dict(), scope=QUENT_SCOPE)


class DequeLogger:
    def __init__(self, deque: collections.deque[dict[str, Any]]):
        self._deque = deque

    def msg(self, **kwargs: Any) -> None:
        with buffer_lock:
            self._deque.append(kwargs)

    # aliases for log levels
    log = debug = info = warn = warning = error = critical = fatal = msg


def collect_to_deque(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    return event_dict


def worker_setup_logging() -> None:
    """
    Setup logging for a single, remote worker.

    Logs are buffered in memory on each worker in a process global. Use
    :func:`drain_buffered_events` to retrieve and clear the buffered events.
    """
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            collect_to_deque,
        ],
        logger_factory=lambda: DequeLogger(log_buffer),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def drain_buffered_events() -> list[dict[str, Any]]:
    """Drain the buffered events."""
    with buffer_lock:
        events = list(log_buffer)
        log_buffer.clear()
    return events

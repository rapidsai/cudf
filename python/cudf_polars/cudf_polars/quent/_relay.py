# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Buffer and relay Quent events from worker processes back to the client.

This module provides framework-agnostic primitives for capturing Quent
telemetry events on remote workers (Dask workers, Ray actors, etc.) and
draining them back to the driver for consolidation.

Usage on a worker process::

    configure_quent_logging()  # call once, early in worker lifecycle
    # ... emit() calls happen normally via quent._logging ...
    events = drain_buffered_events()  # collect and clear the buffer

The structlog processor intercepts only ``scope=QUENT_SCOPE`` records,
so non-Quent log output is unaffected.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import structlog

from cudf_polars.quent._logging import QUENT_SCOPE

if TYPE_CHECKING:
    from collections.abc import MutableMapping

_buffer_lock = threading.Lock()
_event_buffer: list[dict[str, Any]] = []
_configured = False


def _quent_buffer_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Structlog processor that captures QUENT-scoped events into the buffer."""
    if event_dict.get("scope") == QUENT_SCOPE:
        with _buffer_lock:
            _event_buffer.append(dict(event_dict.get("event", event_dict)))
    return event_dict


def configure_quent_logging() -> None:
    """
    Configure structlog on this process to buffer Quent events.

    Idempotent: subsequent calls are no-ops. Must be called before any
    ``emit()`` calls whose events should be captured.
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return
    _configured = True

    structlog.configure(
        processors=[
            _quent_buffer_processor,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def drain_buffered_events() -> list[dict[str, Any]]:
    """
    Return all buffered Quent events and clear the buffer.

    Thread-safe. Intended to be called via ``client.run`` (Dask) or
    as a Ray actor method to relay events back to the driver.
    """
    with _buffer_lock:
        events = list(_event_buffer)
        _event_buffer.clear()
    return events


def get_buffered_events() -> list[dict[str, Any]]:
    """Return a copy of the current buffer without clearing it."""
    with _buffer_lock:
        return list(_event_buffer)

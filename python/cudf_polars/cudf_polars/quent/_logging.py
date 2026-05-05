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


# This buffer is awkward to use. *Especially* during cleanup.
# At the end of the day, we want the client to be able to collect all
# of the events as a flat list[dict]. To enable that, we need to
# gather all the rank-local events and merge.
# Each rank will have logs from their own Worker, and the "client"
# process will also have the client-only events.
#
# How should we design this?


class QuentLogger:
    def __init__(self, maxlen: int = 100_000) -> None:
        self._buffer: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=maxlen
        )
        self._lock = threading.Lock()

    def _get_logger(self, **initial_values: Any) -> Any:
        return structlog.wrap_logger(
            DequeLogger(self._buffer, self._lock),
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                collect_to_deque,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            **initial_values,
        )

    def emit(self, event: Event) -> None:
        logger = self._get_logger()
        logger.info(event.to_dict(), scope=QUENT_SCOPE)

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
        return events

    def collect(self) -> list[dict[str, Any]]:
        return list(self._buffer)


class DequeLogger:
    def __init__(self, deque: collections.deque[dict[str, Any]], lock: threading.Lock):
        self._deque = deque
        self._lock = lock

    def msg(self, **kwargs: Any) -> None:
        with self._lock:
            self._deque.append(kwargs)

    # aliases for log levels
    log = debug = info = warn = warning = error = critical = fatal = msg


def collect_to_deque(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    return event_dict

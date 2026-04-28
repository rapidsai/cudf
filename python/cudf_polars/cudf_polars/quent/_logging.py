# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog  # TODO: optional structlog.

if TYPE_CHECKING:
    from cudf_polars.quent._types import Event

QUENT_SCOPE = "QUENT"


def emit(event: Event) -> None:
    """Emit a Quent event."""
    logger = structlog.get_logger()
    logger.info(event.to_dict(), scope=QUENT_SCOPE)

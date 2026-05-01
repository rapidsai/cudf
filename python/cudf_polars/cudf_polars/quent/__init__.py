# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

from __future__ import annotations

import contextvars

from cudf_polars.quent._context import QuentContext
from cudf_polars.quent._types import Engine, Query, QueryGroup

quent_context = contextvars.ContextVar[QuentContext](
    "quent_context",
    default=QuentContext(),  # noqa: B039
)


__all__ = [
    "Engine",
    "QuentContext",
    "Query",
    "QueryGroup",
    "quent_context",
]

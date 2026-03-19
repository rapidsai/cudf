# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import cudf_polars.dsl.tracing


@pytest.mark.skipif(cudf_polars.dsl.tracing.LOG_TRACES, reason="Tracing is enabled.")
def test_bound_contextvars():
    with cudf_polars.dsl.tracing.bound_contextvars(foo="bar"):
        # no exception is raised
        pass


@pytest.mark.skipif(cudf_polars.dsl.tracing.LOG_TRACES, reason="Tracing is enabled.")
def test_log():
    # no exception is raised
    cudf_polars.dsl.tracing.log("test")

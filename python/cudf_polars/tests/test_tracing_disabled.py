# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import sys

import pytest

import cudf_polars.dsl.tracing


@pytest.fixture
def reload_tracing(monkeypatch):
    # Mock not having structlog installed.
    monkeypatch.setitem(sys.modules, "structlog", None)
    importlib.reload(cudf_polars.dsl.tracing)


@pytest.mark.skipif(cudf_polars.dsl.tracing.LOG_TRACES, reason="Tracing is enabled.")
@pytest.mark.usefixtures("reload_tracing")
def test_bound_contextvars(monkeypatch):
    with cudf_polars.dsl.tracing.bound_contextvars(foo="bar"):
        # no exception is raised
        pass


@pytest.mark.skipif(cudf_polars.dsl.tracing.LOG_TRACES, reason="Tracing is enabled.")
@pytest.mark.usefixtures("reload_tracing")
def test_log():
    # no exception is raised
    cudf_polars.dsl.tracing.log("test")

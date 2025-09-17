# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

import polars as pl

structlog = pytest.importorskip("structlog")
skip_if_not_tracing = pytest.mark.skipif(
    os.environ.get("CUDF_POLARS_LOG_TRACES", "0").lower()
    not in {"1", "true", "y", "yes"},
    reason="Tracing is not enabled",
)


@pytest.fixture(name="log_output")
def fixture_log_output():
    return structlog.testing.LogCapture()


@pytest.fixture(autouse=True)
def fixture_configure_structlog(log_output):
    structlog.configure(processors=[log_output])


@skip_if_not_tracing
def test_trace_basic(
    log_output: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = pl.DataFrame({"a": [1, 2, 3]}).lazy().select(pl.col("a").sum())
    q.collect(engine="gpu")

    assert len(log_output.entries) == 2
    result = dict(log_output.entries[0])
    # pop non-deterministic values
    result.pop("nvml_current_bytes_input")
    result.pop("nvml_current_bytes_output")
    result.pop("start")
    result.pop("stop")

    expected = {
        "count_frames_input": 0,
        "count_frames_output": 1,
        "event": "Execute IR",
        "frames_input": [],
        "frames_output": [{"shape": (3, 1), "size": 24}],
        "log_level": "info",
        "rmm_current_bytes_input": 0,
        "rmm_current_bytes_output": 32,
        "rmm_current_count_input": 0,
        "rmm_current_count_output": 1,
        "rmm_peak_bytes_input": 0,
        "rmm_peak_bytes_output": 32,
        "rmm_peak_count_input": 0,
        "rmm_peak_count_output": 1,
        "rmm_total_bytes_input": 0,
        "rmm_total_bytes_output": 32,
        "rmm_total_count_input": 0,
        "rmm_total_count_output": 1,
        "total_bytes_input": 0,
        "total_bytes_output": 24,
        "type": "DataFrameScan",
    }
    assert result == expected


def test_import_without_structlog(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = list(sys.modules)

    for module in modules:
        if module.startswith("cudf_polars"):
            monkeypatch.delitem(sys.modules, module)
    monkeypatch.setitem(sys.modules, "structlog", None)

    import cudf_polars.dsl.tracing

    assert not cudf_polars.dsl.tracing._HAS_STRUCTLOG

    # And we can run a query without error
    q = pl.DataFrame({"a": [1, 2, 3]}).lazy().select(pl.col("a").sum())
    q.collect(engine="gpu")

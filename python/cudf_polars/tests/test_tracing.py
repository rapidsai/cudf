# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

import cudf_polars.dsl.tracing

structlog = pytest.importorskip("structlog")


@pytest.fixture(name="log_output")
def fixture_log_output():
    return structlog.testing.LogCapture()


@pytest.fixture(autouse=True)
def fixture_configure_structlog(log_output):
    structlog.configure(processors=[log_output])


def test_trace_basic(log_output, monkeypatch) -> None:
    monkeypatch.setattr(cudf_polars.dsl.tracing, "LOG_TRACES", True)
    q = pl.DataFrame({"a": [1, 2, 3]}).lazy().select(pl.col("a").sum())
    q.collect(engine="gpu")
    entries = log_output.entries
    assert len(entries) == 2

    scan, select = entries

    # These fields are variable, so we'll drop them
    scan.pop("start")
    scan.pop("stop")
    scan.pop("rmm_peak_bytes_input")
    scan.pop("rmm_peak_bytes_output")
    scan.pop("rmm_peak_count_input")
    scan.pop("rmm_peak_count_output")
    scan.pop("nvml_current_bytes_input")
    scan.pop("nvml_current_bytes_output")

    assert scan == {
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
        "rmm_total_bytes_input": 0,
        "rmm_total_bytes_output": 32,
        "rmm_total_count_input": 0,
        "rmm_total_count_output": 1,
        "total_bytes_input": 0,
        "total_bytes_output": 24,
        "type": "DataFrameScan",
    }

    assert select["type"] == "Select"
    assert scan["total_bytes_output"] == select["total_bytes_input"]
    assert select["total_bytes_output"] == 8

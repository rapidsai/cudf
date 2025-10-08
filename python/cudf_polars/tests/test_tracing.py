# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import Any

import pytest

import polars as pl

import cudf_polars.testing.asserts

structlog = pytest.importorskip("structlog")


@pytest.fixture(name="log_output")
def fixture_log_output():
    return structlog.testing.LogCapture()


@pytest.fixture(autouse=True)
def fixture_configure_structlog(log_output):
    structlog.configure(processors=[log_output])


def test_trace_basic(
    log_output: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Whether tracing is enabled is determined when cudf_polars is imported.
    # So our best way of testing this is to run things in a subprocess
    # to control the environment and isolate it from the rest of the test suite.
    code = textwrap.dedent("""\
    import polars as pl
    import rmm

    q = pl.DataFrame({"a": [1, 2, 3]}).lazy().select(pl.col("a").sum())
    q.collect(engine=pl.GPUEngine(memory_resource=rmm.mr.ManagedMemoryResource()))
    """)

    env = {
        "CUDF_POLARS__EXECUTOR": cudf_polars.testing.asserts.DEFAULT_EXECUTOR,
        "CUDF_POLARS_LOG_TRACES": "1",
    }

    result = subprocess.check_output([sys.executable, "-c", code], env=env)
    # Just ensure that the default structlog output is in the result
    assert b"Execute IR" in result
    assert b"frames_output" in result
    assert b"frames_input" in result
    assert b"total_bytes_output" in result
    assert b"total_bytes_input" in result
    assert b"rmm_total_bytes_output" in result
    assert b"rmm_total_bytes_input" in result
    assert b"rmm_current_bytes_output" in result
    assert b"overhead_duration" in result


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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import polars as pl

structlog = pytest.importorskip("structlog")


@pytest.fixture(name="log_output")
def fixture_log_output():
    return structlog.testing.LogCapture()


@pytest.fixture(autouse=True)
def fixture_configure_structlog(log_output):
    structlog.configure(processors=[log_output])


def test_import_without_structlog(timeout_seconds: int) -> None:
    # This test could avoid the subprocess by monkeypatching sys.modules, but
    # that was flaky. https://github.com/NVIDIA/cudf/pull/22012#issuecomment-4284536686
    # has more details.
    code = textwrap.dedent("""\
    import sys
    sys.modules["structlog"] = None
    sys.modules["cudf_polars_quent"] = None

    import cudf_polars.dsl.tracing
    assert not cudf_polars.dsl.tracing._HAS_STRUCTLOG

    import polars as pl
    q = pl.DataFrame({"a": [1, 2, 3]}).lazy().select(pl.col("a").sum())
    q.collect(engine="gpu")
    """)
    subprocess.check_call([sys.executable, "-c", code], timeout=timeout_seconds)


@pytest.mark.skipif(
    os.environ.get("CUDF_POLARS_LOG_TRACES") != "1",
    reason="Requires CUDF_POLARS_LOG_TRACES=1.",
)
def test_sets_cudf_polars_query_id():
    pytest.importorskip("cudf_polars_quent")
    left = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    right = pl.LazyFrame({"a": [1, 2, 3], "c": [7, 8, 9]})

    q = left.join(right, on="a", how="inner").select(
        pl.col("b") + pl.col("c").alias("d")
    )
    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
    )

    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as cap:
        q.collect(engine=engine)

    assert len(cap) > 0
    assert "cudf_polars_query_id" in cap[0]
    query_id = cap[0]["cudf_polars_query_id"]

    for log in cap:
        assert "scope" in log
        assert "cudf_polars_query_id" in log
        assert log["cudf_polars_query_id"] == query_id
        keys = set(log.keys())

        assert log["scope"] == "io_task"
        expected_keys = {
            "actor_ir_id",
            "actor_ir_type",
            "admitted",
            "cudf_polars_query_id",
            "estimated_output_bytes",
            "event",
            "ir_id",
            "ir_type",
            "log_level",
            "reservation_bytes",
            "scope",
            "sequence_number",
            "start",
            "stop",
        }
        assert expected_keys.issubset(keys)

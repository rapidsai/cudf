# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import DEFAULT_RUNTIME
from cudf_polars.testing.io import make_lazy_frame


@pytest.mark.parametrize("source_format", ["frame", "parquet", "csv"])
def test_simple_query_with_distributed_support(tmp_path, source_format) -> None:
    # Test a trivial query that works for both the
    # "tasks" and "rapidsmpf" runtimes in distributed mode.

    # Check that we have a distributed cluster running.
    # This tests must be run with: --cluster='distributed'
    distributed = pytest.importorskip("distributed")
    try:
        client = distributed.get_client()
    except ValueError:
        pytest.skip(reason="Requires distributed execution.")

    # check that we have a rapidsmpf cluster running
    pytest.importorskip("rapidsmpf")
    try:
        from rapidsmpf.integrations.dask import bootstrap_dask_cluster

        bootstrap_dask_cluster(client)
    except ValueError:
        pytest.skip(reason="Requires a rapidsmpf-bootstrapped cluster.")

    # Setup the GPUEngine config
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 2,
            "cluster": "distributed",
            "runtime": DEFAULT_RUNTIME,
        },
    )

    # Create a simple DataFrame
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        }
    )

    # Create LazyFrame based on source format
    if source_format == "frame":
        lf = make_lazy_frame(df, fmt="frame")
    else:
        lf = make_lazy_frame(df, fmt=source_format, path=tmp_path, n_files=2)

    # Simple query: select and filter
    q = lf.select("a", "b").filter(pl.col("a") > 2)

    # Should warn about distributed execution being under construction (if rapidsmpf)
    if DEFAULT_RUNTIME == "rapidsmpf":
        with pytest.warns(UserWarning, match="UNDER CONSTRUCTION"):
            result = q.collect(engine=engine)
    else:
        result = q.collect(engine=engine)

    # Check the result is correct
    expected = df.lazy().select("a", "b").filter(pl.col("a") > 2).collect()
    assert result.equals(expected)

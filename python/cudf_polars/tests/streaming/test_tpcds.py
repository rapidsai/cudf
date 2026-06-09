# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS validation tests for the cudf-polars rapidsmpf streaming engine.

Data is expected at the path given by the TPCDS_DATA_DIR environment variable,
with one parquet file per table named <table>.parquet.

Each query is run with the streaming engine and validated against DuckDB on
the same data. Qualification parameters (TPC-DS specification Appendix B) are
used so query parameters are fixed and independent of scale factor.
"""

from __future__ import annotations

import os
import types
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import SPMDEngine
from cudf_polars.streaming.benchmarks.asserts import assert_tpch_result_equal
from cudf_polars.streaming.benchmarks.pdsds import (
    PDSDSDuckDBQueries,
    PDSDSPolarsQueries,
)
from cudf_polars.streaming.benchmarks.utils import (
    POLARS_VALIDATION_OPTIONS,
    execute_duckdb_query,
)

QUERIES = PDSDSPolarsQueries()
DUCKDB_QUERIES = PDSDSDuckDBQueries()
TPCDS_TARGET_PARTITION_SIZE = 2_000_000


@pytest.fixture(scope="session")
def tpcds_data_dir() -> Path:
    data_dir = os.environ.get("TPCDS_DATA_DIR")
    if data_dir is None:
        pytest.skip("TPCDS_DATA_DIR environment variable not set")
    return Path(data_dir)


@pytest.fixture(scope="session")
def tpcds_run_config(tpcds_data_dir: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        dataset_path=tpcds_data_dir,
        suffix=".parquet",
        query_set="pdsds",
        scale_factor=1,
        qualification=True,
    )


@pytest.fixture(scope="session")
def tpcds_engine() -> Generator[SPMDEngine, None, None]:
    options = StreamingOptions(
        target_partition_size=TPCDS_TARGET_PARTITION_SIZE,
        raise_on_fail=True,
        fallback_mode="raise",
        allow_gpu_sharing=True,
    )
    with SPMDEngine.from_options(options) as engine:
        yield engine


@pytest.mark.parametrize(
    "query_id", range(1, 100), ids=[f"q{i}" for i in range(1, 100)]
)
def test_tpcds(
    query_id: int,
    tpcds_run_config: types.SimpleNamespace,
    tpcds_engine: SPMDEngine,
    tpc_iterations: int,
) -> None:
    query_result = getattr(QUERIES, f"q{query_id}")(tpcds_run_config)

    sql = getattr(DUCKDB_QUERIES, f"q{query_id}")(tpcds_run_config)
    expected = execute_duckdb_query(
        sql,
        tpcds_run_config.dataset_path,
        suffix=tpcds_run_config.suffix,
        query_set="pdsds",
    )

    casts = [
        *QUERIES.EXPECTED_CASTS.get(query_id, []),
        *QUERIES.EXPECTED_CASTS_DECIMAL.get(query_id, []),
    ]
    if casts:
        expected = expected.with_columns(*casts)

    for _ in range(tpc_iterations):
        gpu_result = query_result.frame.collect(engine=tpcds_engine)
        assert_tpch_result_equal(
            gpu_result,
            expected,
            sort_by=query_result.sort_by,
            limit=query_result.limit,
            nulls_last=query_result.nulls_last,
            sort_keys=query_result.sort_keys,
            **POLARS_VALIDATION_OPTIONS,
        )

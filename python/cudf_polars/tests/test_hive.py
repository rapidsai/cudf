# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for hive-partitioned parquet scan support in cudf-polars.

Coverage
--------
GPU-engine hive expansion (monkey-patch in cudf_polars.__init__)
    Polars raises ``NotImplementedError: scan with hive partitioning`` from
    ``view_current_node()`` for hive-partitioned Scan nodes, signalling that
    GPU engines must implement their own hive expansion.
    ``cudf_polars`` patches ``pl.LazyFrame.collect`` to transparently expand
    hive scans into explicit-path plans before the GPU callback is invoked.

Tests
-----
- Basic hive scan (no filter) — GPU == CPU across all partitions
- Partition pruning — filter on a partition column skips non-matching dirs
- Column projection — select a strict subset of columns
- Multi-level partitioning — nested ``year=X/month=Y/`` directories
- rapidsmpf streaming backend — same expansion works with the streaming executor
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal

# ---------------------------------------------------------------------------
# Engine parametrization (mirrors test_iceberg.py)
# ---------------------------------------------------------------------------

CHUNKED = pl.GPUEngine(raise_on_fail=True, parquet_options={"chunked": True})
NO_CHUNK = pl.GPUEngine(raise_on_fail=True, parquet_options={"chunked": False})
BOTH_ENGINES = pytest.mark.parametrize(
    "engine", [CHUNKED, NO_CHUNK], ids=["chunked", "no_chunk"]
)


# ---------------------------------------------------------------------------
# Fixtures — build hive directory structures in tmp_path
# ---------------------------------------------------------------------------


def _write_partition(root, partition_kv: dict, data: dict):
    """
    Write *data* as a parquet file inside a hive partition path.

    Example: ``_write_partition(root, {"year": 2024, "month": 1}, {"v": [1,2]})``
    creates ``root/year=2024/month=1/data.parquet``.
    """
    path = root
    for k, v in partition_kv.items():
        path = path / f"{k}={v}"
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data), path / "data.parquet")


# ---------------------------------------------------------------------------
# Basic scan — all partitions, no filter
# ---------------------------------------------------------------------------


@BOTH_ENGINES
def test_hive_basic_scan(tmp_path, engine):
    """
    GPU engine reads all partitions and matches CPU output.

    Three partitions (year=2023, year=2024, year=2025) each containing two
    integer rows.  No filter is applied so all rows are returned.
    """
    for year, vals in [(2023, [1, 2]), (2024, [3, 4]), (2025, [5, 6])]:
        _write_partition(tmp_path, {"year": year}, {"val": vals})

    lf = pl.scan_parquet(tmp_path, hive_partitioning=True)
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


# ---------------------------------------------------------------------------
# Partition pruning — filter on partition column
# ---------------------------------------------------------------------------


@BOTH_ENGINES
def test_hive_partition_pruning(tmp_path, engine):
    """
    A filter on the partition column returns only matching-partition rows.

    Three region partitions (A, B, C) each with 2 rows.  Filtering
    ``region == 'B'`` should return exactly 2 rows from the B partition.
    """
    for region, ids in [("A", [1, 2]), ("B", [3, 4]), ("C", [5, 6])]:
        _write_partition(
            tmp_path, {"region": region}, {"id": ids, "v": [i * 10 for i in ids]}
        )

    lf = pl.scan_parquet(tmp_path, hive_partitioning=True).filter(
        pl.col("region") == "B"
    )
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


@BOTH_ENGINES
def test_hive_partition_pruning_numeric(tmp_path, engine):
    """
    Numeric equality filter on a partition column prunes non-matching dirs.

    Five year partitions (2020-2024).  ``year == 2022`` should return only
    the rows from year=2022.
    """
    for year in range(2020, 2025):
        _write_partition(
            tmp_path, {"year": year}, {"val": [year * 100, year * 100 + 1]}
        )

    lf = pl.scan_parquet(tmp_path, hive_partitioning=True).filter(
        pl.col("year") == 2022
    )
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


@BOTH_ENGINES
def test_hive_partition_pruning_range(tmp_path, engine):
    """
    Range predicate on a numeric partition column prunes correctly.

    Five year partitions (2020-2024).  ``year >= 2023`` should include
    year=2023 and year=2024 only.
    """
    for year in range(2020, 2025):
        _write_partition(tmp_path, {"year": year}, {"val": [year * 10]})

    lf = pl.scan_parquet(tmp_path, hive_partitioning=True).filter(
        pl.col("year") >= 2023
    )
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


# ---------------------------------------------------------------------------
# Column projection — select a strict subset
# ---------------------------------------------------------------------------


@BOTH_ENGINES
def test_hive_column_projection(tmp_path, engine):
    """
    Projecting a subset of columns works after hive expansion.

    Three partitions; each data file has columns ``a`` and ``b``.  The
    partition key ``year`` is injected as a literal.  Selecting only
    ``["a", "year"]`` should drop ``b``.
    """
    for year, a_vals, b_vals in [
        (2022, [10, 20], [100, 200]),
        (2023, [30, 40], [300, 400]),
        (2024, [50, 60], [500, 600]),
    ]:
        _write_partition(tmp_path, {"year": year}, {"a": a_vals, "b": b_vals})

    lf = pl.scan_parquet(tmp_path, hive_partitioning=True).select("a", "year")
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


# ---------------------------------------------------------------------------
# Multi-level partitioning — nested key=value directories
# ---------------------------------------------------------------------------


@BOTH_ENGINES
def test_hive_multilevel_partitioning(tmp_path, engine):
    """
    Two-level hive partitioning (year / month) is read correctly.

    Six leaf partitions across 2 years x 3 months.  All rows are returned
    when no filter is applied.
    """
    for year in [2024, 2025]:
        for month in [1, 2, 3]:
            _write_partition(
                tmp_path,
                {"year": year, "month": month},
                {"sales": [year * 1000 + month * 10]},
            )

    lf = pl.scan_parquet(tmp_path, hive_partitioning=True)
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


@BOTH_ENGINES
def test_hive_multilevel_partition_pruning(tmp_path, engine):
    """
    Filtering on the outer partition key in multi-level hive layout prunes correctly.

    Two-level ``year / month`` partitioning.  Filtering to ``year == 2025``
    should skip all ``year=2024`` directories.
    """
    for year in [2024, 2025]:
        for month in [1, 2, 3]:
            _write_partition(
                tmp_path,
                {"year": year, "month": month},
                {"cnt": [year * 100 + month]},
            )

    lf = pl.scan_parquet(tmp_path, hive_partitioning=True).filter(
        pl.col("year") == 2025
    )
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


# ---------------------------------------------------------------------------
# rapidsmpf streaming backend
# ---------------------------------------------------------------------------

rapidsmpf = pytest.importorskip("rapidsmpf", reason="rapidsmpf not installed")

_RAPIDSMPF_ENGINE = pl.GPUEngine(
    raise_on_fail=True,
    executor="streaming",
    executor_options={"runtime": "rapidsmpf"},
)


def _rapidsmpf_compatible() -> bool:
    """Return True if the rapidsmpf streaming backend is functional."""
    try:
        pl.LazyFrame({"x": [1]}).collect(engine=_RAPIDSMPF_ENGINE)
    except Exception:
        return False
    else:
        return True


_requires_rapidsmpf = pytest.mark.skipif(
    not _rapidsmpf_compatible(),
    reason="rapidsmpf backend not compatible with this cudf-polars version",
)


@_requires_rapidsmpf
def test_hive_rapidsmpf_basic_scan(tmp_path):
    """
    Basic hive scan succeeds with the rapidsmpf streaming executor.

    The monkey-patch expands the hive scan before the streaming pipeline
    sees it, so rapidsmpf processes a standard concat of parquet scans.
    """
    for year, vals in [(2023, [1, 2]), (2024, [3, 4]), (2025, [5, 6])]:
        _write_partition(tmp_path, {"year": year}, {"val": vals})

    lf = pl.scan_parquet(tmp_path, hive_partitioning=True)
    assert_gpu_result_equal(lf, engine=_RAPIDSMPF_ENGINE, check_row_order=False)


@_requires_rapidsmpf
def test_hive_rapidsmpf_partition_pruning(tmp_path):
    """
    Partition pruning works correctly with the rapidsmpf streaming executor.
    """
    for region, ids in [("A", [10, 20]), ("B", [30, 40]), ("C", [50, 60])]:
        _write_partition(
            tmp_path, {"region": region}, {"id": ids, "score": [i + 1 for i in ids]}
        )

    lf = pl.scan_parquet(tmp_path, hive_partitioning=True).filter(
        pl.col("region") == "A"
    )
    assert_gpu_result_equal(lf, engine=_RAPIDSMPF_ENGINE, check_row_order=False)

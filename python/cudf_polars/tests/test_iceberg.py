# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Iceberg scan support in cudf-polars.

Coverage
--------
End-to-end Iceberg scans (PyIceberg + native reader)
    Build real Iceberg V1 tables on-disk via PyIceberg, force the native
    Polars reader (``POLARS_ICEBERG_READER_OVERRIDE=native``), and compare
    GPU vs CPU results via ``assert_gpu_result_equal``.

All tests require a CUDA-capable GPU (standard for this test suite).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_131

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

pyiceberg = pytest.importorskip("pyiceberg", reason="pyiceberg not installed")
sqlalchemy = pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")

# PyIceberg's SqlCatalog leaves SQLite connections open until GC finalises
# them.  Python 3.14 raises ResourceWarning more aggressively in this case;
# suppress it for the in-process throwaway catalogs used in these tests.
pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning")

CHUNKED = pl.GPUEngine(raise_on_fail=True, parquet_options={"chunked": True})
NO_CHUNK = pl.GPUEngine(raise_on_fail=True, parquet_options={"chunked": False})
BOTH_ENGINES = pytest.mark.parametrize(
    "engine", [CHUNKED, NO_CHUNK], ids=["chunked", "no_chunk"]
)


def _make_catalog(tmp_path: Path, name: str = "test"):
    """Create a SQLite-backed in-process PyIceberg catalog."""
    from pyiceberg.catalog.sql import SqlCatalog

    cat = SqlCatalog(
        name,
        uri=f"sqlite:///{tmp_path / 'catalog.db'}",
        warehouse=str(tmp_path),
    )
    cat.create_namespace("ns")
    return cat


@pytest.fixture(autouse=False)
def native_iceberg_reader(monkeypatch):
    """Force the native Iceberg reader so Polars emits Scan, not PythonScan."""
    monkeypatch.setenv("POLARS_ICEBERG_READER_OVERRIDE", "native")


# ---------------------------------------------------------------------------
# _reconcile_schema via plain scan_parquet (no Iceberg dependency)
# ---------------------------------------------------------------------------

# TODO: re-enable schema-evolution tests once the streaming executor's
# statistics phase (experimental/io.py ParquetSourceInfo.__init__) can handle
# parquet files with heterogeneous column counts in the same scan.
# The streaming executor calls plc.io.parquet_metadata.read_parquet_metadata
# on all paths as a single batch; libcudf rejects the batch when column counts
# differ across files ("All non-empty sources must have the same number of
# columns").  This is a pre-existing gap in experimental/io.py — not caused by
# the iceberg changes — but it blocks:
#   - test_reconcile_missing_column_inserted_as_null
#   - test_reconcile_missing_column_with_predicate
#   - test_iceberg_schema_evolution_missing_column
#   - test_iceberg_schema_evolution_with_filter
# Our TPC-H data has no schema evolution so the integration test is unaffected.


@BOTH_ENGINES
def test_reconcile_column_order_normalised(tmp_path, engine):
    """
    Files with columns in different order than the projected schema.

    After reading, ``_reconcile_schema`` must reorder to match the schema
    so column-order-sensitive downstream operations are correct.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table({"b": [1, 2], "a": [3, 4]}),
        tmp_path / "f.parquet",
    )

    # Project in a different order — schema will be [a, b]
    lf = pl.scan_parquet(tmp_path / "f.parquet").select("a", "b")
    assert_gpu_result_equal(lf, engine=engine)


# ---------------------------------------------------------------------------
# End-to-end Iceberg scans
# ---------------------------------------------------------------------------


@BOTH_ENGINES
def test_iceberg_basic_scan(tmp_path, native_iceberg_reader, engine):
    """
    Basic V1 Iceberg table: no schema evolution, no deletes.

    Validates the GPU path through the native reader for a simple
    three-column mixed-type table.
    """
    import pyarrow as pa
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType

    cat = _make_catalog(tmp_path)
    ice_schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "name", StringType(), required=False),
        NestedField(3, "val", DoubleType(), required=False),
    )
    tbl = cat.create_table("ns.basic", schema=ice_schema)
    tbl.append(
        pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"], "val": [1.1, 2.2, 3.3]})
    )

    lf = pl.scan_iceberg(tbl.metadata_location)
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


@BOTH_ENGINES
def test_iceberg_scan_with_predicate(tmp_path, native_iceberg_reader, engine):
    """Predicate pushdown through native Iceberg scan produces correct results."""
    import pyarrow as pa
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    cat = _make_catalog(tmp_path)
    ice_schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "val", LongType(), required=False),
    )
    tbl = cat.create_table("ns.pred", schema=ice_schema)
    tbl.append(pa.table({"id": list(range(10)), "val": list(range(10, 20))}))

    lf = pl.scan_iceberg(tbl.metadata_location).filter(pl.col("id") > 5)
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


@BOTH_ENGINES
def test_iceberg_partition_pruning(tmp_path, native_iceberg_reader, engine):
    """
    Partitioned Iceberg table: a filter on the partition column prunes
    whole parquet files before the GPU reader ever sees them.

    Three identity partitions (region A / B / C) are written as separate
    appends, producing three distinct data files.  Filtering to
    ``region == 'A'`` should reduce the file list from 3 → 1 at Polars
    optimizer time (native reader reads Iceberg manifests and bakes the
    pruned paths into the ``Scan`` node).

    Verifies:
    - PyIceberg confirms pruning at the metadata level (sanity check).
    - CPU and GPU (cudf-polars) produce identical results.
    """
    import pyarrow as pa
    from pyiceberg.expressions import EqualTo
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import LongType, NestedField, StringType

    cat = _make_catalog(tmp_path)
    ice_schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "region", StringType(), required=False),
        NestedField(3, "val", LongType(), required=False),
    )
    partition_spec = PartitionSpec(
        PartitionField(
            source_id=2,
            field_id=1000,
            transform=IdentityTransform(),
            name="region",
        )
    )
    tbl = cat.create_table(
        "ns.partitioned", schema=ice_schema, partition_spec=partition_spec
    )

    # Three separate appends → three distinct partition data files
    tbl.append(pa.table({"id": [1, 2], "region": ["A", "A"], "val": [10, 20]}))
    tbl.append(pa.table({"id": [3, 4], "region": ["B", "B"], "val": [30, 40]}))
    tbl.append(pa.table({"id": [5, 6], "region": ["C", "C"], "val": [50, 60]}))

    # Confirm Iceberg metadata agrees that the filter prunes files
    all_files = list(tbl.scan().plan_files())
    pruned_files = list(tbl.scan(row_filter=EqualTo("region", "A")).plan_files())
    assert len(pruned_files) < len(all_files), (
        f"Expected partition pruning to reduce file count; "
        f"got {len(pruned_files)} / {len(all_files)}"
    )

    lf = pl.scan_iceberg(tbl.metadata_location).filter(pl.col("region") == "A")
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


# ---------------------------------------------------------------------------
# Integration test — real TPC-H scale-1000 data
# ---------------------------------------------------------------------------

_SCALE1000_ICEBERG = Path("/datasets/nperera/scale-1000-iceberg")
_SCALE1000_CATALOG = _SCALE1000_ICEBERG / "catalog.db"

_requires_scale1000 = pytest.mark.skipif(
    not _SCALE1000_CATALOG.exists(),
    reason=(
        "scale-1000 Iceberg data not present; "
        "run python/cudf_polars/tests/make_tpch_iceberg.py --tables orders first"
    ),
)


@_requires_scale1000
@BOTH_ENGINES
def test_iceberg_integration_orders_partition_pruning(native_iceberg_reader, engine):
    """
    Integration test against a real TPC-H scale-1000 orders Iceberg table.

    The orders table is monthly-partitioned on ``o_orderdate`` spanning
    1992-01-01 → 1998-12-31 (~84 monthly partitions).  Filtering to
    ``o_orderdate < date(1992, 3, 1)`` leaves ≤ 2 partitions, pruning the
    vast majority of data files.

    Requires the Iceberg layer to be built first::

        python python/cudf_polars/tests/make_tpch_iceberg.py --tables orders

    Verifies:
    - PyIceberg confirms the filter prunes files at the metadata level.
    - CPU and GPU (cudf-polars) produce identical aggregation results.
    """
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.expressions import LessThan

    catalog = SqlCatalog(
        "tpch",
        uri=f"sqlite:///{_SCALE1000_CATALOG}",
        warehouse=f"file://{_SCALE1000_ICEBERG}",
    )
    ice_tbl = catalog.load_table("tpch.orders")

    # Confirm partition pruning at the Iceberg metadata level
    cutoff = "1992-03-01"
    all_files = list(ice_tbl.scan().plan_files())
    pruned_files = list(
        ice_tbl.scan(row_filter=LessThan("o_orderdate", cutoff)).plan_files()
    )
    assert len(pruned_files) < len(all_files), (
        f"Expected partition pruning: got {len(pruned_files)} / {len(all_files)} files"
    )

    lf = (
        pl.scan_iceberg(ice_tbl.metadata_location)
        .filter(pl.col("o_orderdate") < pl.date(1992, 3, 1))
        .group_by("o_orderstatus")
        .agg(
            pl.len().alias("cnt"),
            pl.col("o_totalprice").sum().alias("sum"),
        )
        .sort("o_orderstatus")
    )
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


@BOTH_ENGINES
def test_iceberg_column_projection(tmp_path, native_iceberg_reader, engine):
    """Column projection selects a strict subset of the Iceberg schema columns."""
    import pyarrow as pa
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType

    cat = _make_catalog(tmp_path)
    ice_schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "name", StringType(), required=False),
        NestedField(3, "val", DoubleType(), required=False),
    )
    tbl = cat.create_table("ns.proj", schema=ice_schema)
    tbl.append(
        pa.table({"id": [1, 2, 3], "name": ["x", "y", "z"], "val": [0.1, 0.2, 0.3]})
    )

    lf = pl.scan_iceberg(tbl.metadata_location).select("id", "val")
    assert_gpu_result_equal(lf, engine=engine, check_row_order=False)


@pytest.mark.skipif(
    POLARS_VERSION_LT_131,
    reason="deletion_files attribute added in Polars 1.31",
)
def test_iceberg_deletion_files_raises(tmp_path, native_iceberg_reader):
    """
    Iceberg V2 position-delete files must raise NotImplementedError at
    translation time so the caller knows to fall back rather than silently
    returning stale rows.

    Requires a PyIceberg version that supports merge-on-read deletes.
    PyIceberg 0.11.x falls back to copy-on-write, producing no delete files;
    this test is skipped automatically when that fallback is detected.
    """
    import warnings

    import pyarrow as pa
    from pyiceberg.expressions import EqualTo
    from pyiceberg.schema import Schema
    from pyiceberg.table import TableProperties
    from pyiceberg.types import LongType, NestedField

    cat = _make_catalog(tmp_path)
    ice_schema = Schema(
        NestedField(1, "id", LongType(), required=False),
    )
    tbl = cat.create_table(
        "ns.deletes",
        schema=ice_schema,
        properties={
            TableProperties.DELETE_MODE: TableProperties.DELETE_MODE_MERGE_ON_READ,
        },
    )
    tbl.append(pa.table({"id": [1, 2, 3]}))

    # Attempt a delete; PyIceberg may warn and fall back to copy-on-write.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tbl.delete(EqualTo("id", 2))

    if any("copy-on-write" in str(w.message).lower() for w in caught):
        pytest.skip(
            "PyIceberg fell back to copy-on-write — merge-on-read deletes are "
            "not supported by this PyIceberg version; skipping."
        )

    # Verify delete files were actually produced
    has_delete_files = any(f.delete_files for f in tbl.scan().plan_files())
    if not has_delete_files:
        pytest.skip("No position-delete files produced; skipping.")

    lf = pl.scan_iceberg(tbl.metadata_location)
    assert_ir_translation_raises(lf, NotImplementedError)

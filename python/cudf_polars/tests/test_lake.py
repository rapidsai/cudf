# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import Any, Literal

    from pyiceberg.table.update.schema import UpdateSchema

pytest.importorskip("pyiceberg")
deltalake = pytest.importorskip("deltalake")
import pyiceberg.schema  # noqa: E402
from pyiceberg.catalog.sql import SqlCatalog  # noqa: E402
from pyiceberg.partitioning import PartitionField, PartitionSpec  # noqa: E402
from pyiceberg.transforms import IdentityTransform  # noqa: E402
from pyiceberg.types import (  # noqa: E402
    DecimalType,
    DoubleType,
    ListType,
    LongType,
    MapType,
    NestedField,
    StringType,
    StructType,
)


@pytest.fixture
def iceberg_catalog(tmp_path: Path) -> Iterator[SqlCatalog]:
    with SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=f"file://{tmp_path / 'warehouse'}",
    ) as catalog:
        catalog.create_namespace_if_not_exists("ns")
        yield catalog


@pytest.fixture
def flat_iceberg(iceberg_catalog: SqlCatalog) -> str:
    first = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tbl = iceberg_catalog.create_table(
        "ns.flat", schema=first.collect_schema().to_arrow()
    )
    first.sink_iceberg(tbl, mode="append")
    pl.LazyFrame({"a": [4, 5], "b": ["p", "q"]}).sink_iceberg(tbl, mode="append")
    return tbl.metadata_location


@pytest.fixture
def evolved_iceberg(iceberg_catalog: SqlCatalog) -> str:
    first = pl.LazyFrame(
        {"a": pl.Series([1, 2, 3], dtype=pl.Int32), "b": ["x", "y", "z"]}
    )
    tbl = iceberg_catalog.create_table(
        "ns.evolved", schema=first.collect_schema().to_arrow()
    )
    first.sink_iceberg(tbl, mode="append")

    with tbl.update_schema() as update:
        update.rename_column("b", "b_renamed")
    tbl.refresh()

    pl.LazyFrame(
        {
            "a": pl.Series([4, 5], dtype=pl.Int64),
            "b_renamed": ["p", "q"],
            "c": [1.5, 2.5],
        }
    ).sink_iceberg(tbl, mode="append", schema_mode="merge")
    return tbl.metadata_location


@pytest.fixture
def renamed_iceberg(iceberg_catalog: SqlCatalog) -> str:
    first = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tbl = iceberg_catalog.create_table(
        "ns.renamed", schema=first.collect_schema().to_arrow()
    )
    first.sink_iceberg(tbl, mode="append")

    with tbl.update_schema() as update:
        update.rename_column("b", "b_renamed")
    tbl.refresh()

    pl.LazyFrame({"a": [4, 5], "b_renamed": ["p", "q"]}).sink_iceberg(
        tbl, mode="append"
    )
    return tbl.metadata_location


@pytest.fixture
def promoted_iceberg(iceberg_catalog: SqlCatalog) -> str:
    first = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, 3], dtype=pl.Int32),
            "b": ["x", "y", "z"],
            "c": pl.Series([0.5, 1.5, 2.5], dtype=pl.Float32),
        }
    )
    tbl = iceberg_catalog.create_table(
        "ns.promoted", schema=first.collect_schema().to_arrow()
    )
    first.sink_iceberg(tbl, mode="append")

    with tbl.update_schema() as update:
        update.update_column("a", field_type=LongType())
        update.update_column("c", field_type=DoubleType())
    tbl.refresh()

    pl.LazyFrame(
        {
            "a": pl.Series([4, 5], dtype=pl.Int64),
            "b": ["p", "q"],
            "c": pl.Series([3.5, 4.5], dtype=pl.Float64),
        }
    ).sink_iceberg(tbl, mode="append")
    return tbl.metadata_location


@pytest.fixture
def partitioned_iceberg(iceberg_catalog: SqlCatalog) -> str:
    schema = pyiceberg.schema.Schema(
        NestedField(1, "a", LongType(), required=False),
        NestedField(2, "part", StringType(), required=False),
    )
    spec = PartitionSpec(
        PartitionField(
            source_id=2, field_id=1000, transform=IdentityTransform(), name="part"
        )
    )
    tbl = iceberg_catalog.create_table(
        "ns.partitioned", schema=schema, partition_spec=spec
    )
    pl.LazyFrame({"a": [1, 2, 3, 4], "part": ["u", "u", "v", "v"]}).sink_iceberg(
        tbl, mode="append"
    )
    return tbl.metadata_location


@pytest.fixture
def flat_delta(tmp_path: Path) -> str:
    path = tmp_path / "delta_flat"
    pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).sink_delta(
        str(path), mode="overwrite"
    )
    pl.LazyFrame({"a": [4, 5], "b": ["p", "q"]}).sink_delta(str(path), mode="append")
    return str(path)


@pytest.fixture
def deleted_rows_delta(tmp_path: Path) -> str:
    path = tmp_path / "delta_deleted"
    pl.LazyFrame({"a": list(range(10)), "b": [f"s{i}" for i in range(10)]}).sink_delta(
        str(path),
        mode="overwrite",
        delta_write_options={"configuration": {"delta.enableDeletionVectors": "true"}},
    )
    deltalake.DeltaTable(str(path)).delete("a % 3 == 0")
    return str(path)


@pytest.fixture
def partitioned_delta(tmp_path: Path) -> str:
    path = tmp_path / "delta_partitioned"
    pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5, 6], "part": ["u", "u", "v", "v", "w", "w"]}
    ).sink_delta(
        str(path),
        mode="overwrite",
        delta_write_options={"partition_by": ["part"]},
    )
    return str(path)


@pytest.mark.parametrize(
    "table",
    [
        "flat_iceberg",
        "flat_delta",
        "evolved_iceberg",
        "renamed_iceberg",
        "promoted_iceberg",
        "partitioned_iceberg",
        "partitioned_delta",
        "deleted_rows_delta",
    ],
)
def test_scan_lake(
    request: pytest.FixtureRequest, engine: pl.GPUEngine, table: str
) -> None:
    scan = pl.scan_iceberg if table.endswith("iceberg") else pl.scan_delta
    q = scan(request.getfixturevalue(table))
    assert_gpu_result_equal(q.sort("a"), engine=engine)


FLAT_PREDICATES = [
    pl.col("a") < 3,
    pl.col("b") == "q",
    (pl.col("a") > 1) & (pl.col("b") != "y"),
    pl.col("a").is_between(2, 4),
]
EVOLVED_PREDICATES = [
    pl.col("a") > 2,
    pl.col("b_renamed") == "y",
    pl.col("b_renamed").is_in(["x", "q"]),
    pl.col("c") > 2.0,
    pl.col("c").is_null(),
    (pl.col("b_renamed") != "x") & (pl.col("a") < 5),
    (pl.col("a") > 1) | (pl.col("c") > 2.0),
]
PROMOTED_PREDICATES = [
    pl.col("a") > 2,
    pl.col("c") < 3.0,
    (pl.col("a") > 1) & (pl.col("b") != "q"),
]
PARTITIONED_PREDICATES = [
    pl.col("a") > 1,
    pl.col("a").is_in([2, 3, 6]),
    pl.col("part") == "v",
    (pl.col("a") > 1) & (pl.col("part") == "v"),
    (pl.col("a") == 1) | (pl.col("part") == "v"),
]


@pytest.mark.parametrize(
    "table, predicate",
    [
        *(
            (table, p)
            for table in ("flat_iceberg", "flat_delta")
            for p in FLAT_PREDICATES
        ),
        *(("evolved_iceberg", p) for p in EVOLVED_PREDICATES),
        ("renamed_iceberg", pl.col("b_renamed") != "y"),
        *(("promoted_iceberg", p) for p in PROMOTED_PREDICATES),
        *(
            (table, p)
            for table in ("partitioned_iceberg", "partitioned_delta")
            for p in PARTITIONED_PREDICATES
        ),
        ("deleted_rows_delta", pl.col("a") > 4),
    ],
)
def test_scan_lake_predicate(
    request: pytest.FixtureRequest,
    engine: pl.GPUEngine,
    table: str,
    predicate: pl.Expr,
) -> None:
    scan = pl.scan_iceberg if table.endswith("iceberg") else pl.scan_delta
    q = scan(request.getfixturevalue(table))
    assert_gpu_result_equal(q.filter(predicate).sort("a"), engine=engine)


@pytest.mark.parametrize(
    "query",
    [
        *(
            lambda q, columns=columns: q.select(columns).sort(columns)
            for columns in (
                ["a"],
                ["b_renamed"],
                ["c"],
                ["c", "a"],
                ["a", "b_renamed", "c"],
            )
        ),
        *(lambda q, n=n: q.sort("a").head(n) for n in (1, 3, 5)),
        lambda q: q.with_row_index("idx"),
    ],
    ids=[
        "select_a",
        "select_b_renamed",
        "select_c",
        "select_c_a",
        "select_all",
        "head_1",
        "head_3",
        "head_5",
        "row_index",
    ],
)
def test_scan_iceberg_schema_evolution(
    engine: pl.GPUEngine, evolved_iceberg: str, query: Any
) -> None:
    assert_gpu_result_equal(query(pl.scan_iceberg(evolved_iceberg)), engine=engine)


@pytest.mark.parametrize(
    "query",
    [
        lambda q: q.filter(pl.col("b_renamed") == "y").with_row_index("idx"),
        lambda q: q.filter(pl.col("b_renamed") == "y").head(1),
        lambda q: q.with_row_index("idx").filter(pl.col("b_renamed") == "y"),
        lambda q: q.head(4).filter(pl.col("b_renamed") == "y"),
    ],
    ids=[
        "filter_then_row_index",
        "filter_then_slice",
        "row_index_then_filter",
        "slice_then_filter",
    ],
)
def test_scan_lake_predicate_pushdown(
    in_memory_engine: pl.GPUEngine, evolved_iceberg: str, query: Any
) -> None:
    assert_gpu_result_equal(
        query(pl.scan_iceberg(evolved_iceberg)),
        engine=in_memory_engine,
        check_row_order=False,
    )


def test_scan_iceberg_partition_predicate_without_file_columns(
    engine: pl.GPUEngine, partitioned_iceberg: str
) -> None:
    q = pl.scan_iceberg(partitioned_iceberg).filter(pl.col("part") != "w")
    assert_gpu_result_equal(q.select("part"), engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "query",
    [
        lambda q: q.filter(pl.col("a") > 2),
        lambda q: q.filter(pl.col("b") > 30),
        lambda q: q.filter(pl.col("b") > 30).select("b"),
        lambda q: q.filter(pl.col("a").is_in([1, 4, 6])).select("part"),
        lambda q: q.filter((pl.col("a") < 5) & (pl.col("part") != "v")),
        lambda q: q.select("part"),
    ],
    ids=["data", "added", "added_only", "partition_only", "mixed", "partitions"],
)
def test_scan_delta_evolved_partitions_predicate(
    engine: pl.GPUEngine, tmp_path: Path, query: Any
) -> None:
    path = str(tmp_path / "delta")
    pl.LazyFrame({"a": [1, 2, 3, 4], "part": ["u", "u", "v", "v"]}).sink_delta(
        path, mode="overwrite", delta_write_options={"partition_by": ["part"]}
    )
    pl.LazyFrame({"a": [5, 6], "b": [50, 60], "part": ["w", "w"]}).sink_delta(
        path, mode="append", delta_write_options={"schema_mode": "merge"}
    )
    assert_gpu_result_equal(
        query(pl.scan_delta(path)), engine=engine, check_row_order=False
    )


@pytest.mark.parametrize(
    "first, second",
    [
        (
            (
                {"s": [{"x": 1, "y": 2}, None]},
                {"s": pl.Struct({"x": pl.Int64, "y": pl.Int64})},
            ),
            (
                {"s": [{"y": 4, "x": 3}]},
                {"s": pl.Struct({"y": pl.Int64, "x": pl.Int64})},
            ),
        ),
        (
            ({"s": [{"x": 1}, None]}, {"s": pl.Struct({"x": pl.Int64})}),
            (
                {"s": [{"x": 3, "t": {"u": "p"}, "l": [1, 2]}]},
                {
                    "s": pl.Struct(
                        {
                            "x": pl.Int64,
                            "t": pl.Struct({"u": pl.String}),
                            "l": pl.List(pl.Int64),
                        }
                    )
                },
            ),
        ),
        (
            ({"a": [1]}, {"a": pl.Int64}),
            (
                {"a": [2], "s": [{"x": 3}]},
                {"a": pl.Int64, "s": pl.Struct({"x": pl.Int64})},
            ),
        ),
        (
            (
                {"l": [[{"x": 1, "y": 2}], None, []]},
                {"l": pl.List(pl.Struct({"x": pl.Int64, "y": pl.Int64}))},
            ),
            (
                {"l": [[{"y": 4, "x": 3}]]},
                {"l": pl.List(pl.Struct({"y": pl.Int64, "x": pl.Int64}))},
            ),
        ),
    ],
    ids=[
        "reordered_fields",
        "added_nested_fields",
        "added_struct_column",
        "list_of_structs",
    ],
)
def test_scan_delta_nested_schema_evolution(
    engine: pl.GPUEngine,
    tmp_path: Path,
    first: tuple[dict[str, Any], dict[str, pl.DataType]],
    second: tuple[dict[str, Any], dict[str, pl.DataType]],
) -> None:
    path = str(tmp_path / "delta")
    pl.LazyFrame(*first).sink_delta(path, mode="overwrite")
    pl.LazyFrame(*second).sink_delta(
        path, mode="append", delta_write_options={"schema_mode": "merge"}
    )
    assert_gpu_result_equal(pl.scan_delta(path), engine=engine, check_row_order=False)


def test_scan_iceberg_readded_column(
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.readded",
        schema=pyiceberg.schema.Schema(
            NestedField(1, "a", LongType(), required=False),
            NestedField(2, "b", LongType(), required=False),
        ),
    )
    tbl.append(
        pa.Table.from_pylist([{"a": 1, "b": 10}], schema=tbl.schema().as_arrow())
    )
    with tbl.update_schema() as update:
        update.delete_column("a")
    with tbl.update_schema() as update:
        update.add_column("a", LongType())
    tbl.append(
        pa.Table.from_pylist([{"a": 2, "b": 20}], schema=tbl.schema().as_arrow())
    )
    assert_gpu_result_equal(
        pl.scan_iceberg(tbl.metadata_location), engine=engine, check_row_order=False
    )


def test_scan_iceberg_without_field_ids(
    in_memory_engine: pl.GPUEngine, iceberg_catalog: SqlCatalog, tmp_path: Path
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.added_files",
        schema=pyiceberg.schema.Schema(NestedField(1, "a", LongType(), required=False)),
    )
    tbl.append(pa.Table.from_pylist([{"a": 1}], schema=tbl.schema().as_arrow()))
    path = str(tmp_path / "without_field_ids.parquet")
    pq.write_table(pa.table({"a": pa.array([2], pa.int64())}), path)
    tbl.add_files([path])
    q = pl.scan_iceberg(tbl.metadata_location)
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=re.escape(
            "NotImplementedError: Iceberg data file without parquet field IDs"
        ),
    ):
        q.collect(engine=in_memory_engine)


@pytest.mark.parametrize(
    "kind, first, evolutions, second",
    [
        (
            "struct",
            [{"s": {"x": 1, "y": 2}}, {"s": None}],
            (lambda update: update.rename_column("s.y", "z"),),
            [{"s": {"x": 3, "z": 4}}],
        ),
        (
            "struct",
            [{"s": {"x": 1, "y": 2}}, {"s": None}],
            (
                lambda update: update.rename_column("s.x", "tmp"),
                lambda update: update.rename_column("s.y", "x"),
                lambda update: update.rename_column("s.tmp", "y"),
            ),
            [{"s": {"y": 3, "x": 4}}],
        ),
        (
            "struct",
            [{"s": {"x": 1, "y": 2}}, {"s": None}],
            (
                lambda update: update.delete_column("s.y"),
                lambda update: update.add_column(("s", "y"), LongType()),
            ),
            [{"s": {"x": 3, "y": 4}}],
        ),
        (
            "list",
            [{"s": [{"u": 1}]}, {"s": None}, {"s": []}],
            (lambda update: update.rename_column("s.element.u", "v"),),
            [{"s": [{"v": 2}, None]}],
        ),
        (
            "string_list",
            [{"s": {"p": ["a", None]}}, {"s": None}],
            (lambda update: update.rename_column("s.p", "q"),),
            [{"s": {"q": []}}, {"s": {"q": ["b"]}}, {"s": {"q": None}}],
        ),
    ],
    ids=[
        "renamed_field",
        "swapped_names",
        "readded_field",
        "list_element_field",
        "string_list_field",
    ],
)
def test_scan_iceberg_nested_by_field_id(
    engine: pl.GPUEngine,
    iceberg_catalog: SqlCatalog,
    kind: Literal["struct", "list", "string_list"],
    first: list[dict[str, Any]],
    evolutions: tuple[Callable[[UpdateSchema], Any], ...],
    second: list[dict[str, Any]],
) -> None:
    if kind == "struct":
        column: StructType | ListType = StructType(
            NestedField(2, "x", LongType(), required=False),
            NestedField(3, "y", LongType(), required=False),
        )
    elif kind == "list":
        column = ListType(
            2,
            StructType(NestedField(3, "u", LongType(), required=False)),
            element_required=False,
        )
    else:
        column = StructType(
            NestedField(
                2,
                "p",
                ListType(3, StringType(), element_required=False),
                required=False,
            )
        )
    tbl = iceberg_catalog.create_table(
        "ns.nested",
        schema=pyiceberg.schema.Schema(NestedField(1, "s", column, required=False)),
    )
    tbl.append(pa.Table.from_pylist(first, schema=tbl.schema().as_arrow()))
    for evolve in evolutions:
        with tbl.update_schema() as update:
            evolve(update)
    tbl.append(pa.Table.from_pylist(second, schema=tbl.schema().as_arrow()))
    assert_gpu_result_equal(
        pl.scan_iceberg(tbl.metadata_location), engine=engine, check_row_order=False
    )


def test_scan_iceberg_nested_by_field_id_filter(
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.nested",
        schema=pyiceberg.schema.Schema(
            NestedField(
                1,
                "s",
                StructType(
                    NestedField(2, "x", LongType(), required=False),
                    NestedField(3, "y", LongType(), required=False),
                ),
                required=False,
            )
        ),
    )
    tbl.append(
        pa.Table.from_pylist(
            [{"s": {"x": 1, "y": 2}}, {"s": None}], schema=tbl.schema().as_arrow()
        )
    )
    with tbl.update_schema() as update:
        update.rename_column("s.y", "z")
    tbl.append(
        pa.Table.from_pylist([{"s": {"x": 3, "z": 4}}], schema=tbl.schema().as_arrow())
    )
    q = pl.scan_iceberg(tbl.metadata_location).filter(pl.col("s").struct.field("z") > 1)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_scan_iceberg_dropped_map_column(
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.dropped_map",
        schema=pyiceberg.schema.Schema(
            NestedField(
                1,
                "m",
                MapType(2, StringType(), 3, LongType(), value_required=False),
                required=False,
            ),
            NestedField(4, "t", LongType(), required=False),
        ),
    )
    tbl.append(
        pa.Table.from_pylist(
            [{"m": [("k", 1)], "t": 1}], schema=tbl.schema().as_arrow()
        )
    )
    with tbl.update_schema() as update:
        update.delete_column("m")
    tbl.append(pa.Table.from_pylist([{"t": 2}], schema=tbl.schema().as_arrow()))
    assert_gpu_result_equal(
        pl.scan_iceberg(tbl.metadata_location), engine=engine, check_row_order=False
    )


@pytest.mark.parametrize(
    "column, projection, match",
    [
        ("map", "t", "Iceberg column mapping of map columns"),
        ("map_in_struct", "t", "Iceberg column mapping of map columns"),
        ("nested_partition", "s", "Iceberg partition values for nested fields"),
    ],
)
def test_scan_iceberg_nested_unsupported(
    in_memory_engine: pl.GPUEngine,
    iceberg_catalog: SqlCatalog,
    column: Literal["map", "map_in_struct", "nested_partition"],
    projection: str,
    match: str,
) -> None:
    if column == "map":
        field_type: MapType | StructType = MapType(
            2, StringType(), 3, LongType(), value_required=False
        )
        row = {"s": [("k", 1)], "t": 1}
    elif column == "map_in_struct":
        field_type = StructType(
            NestedField(
                2,
                "m",
                MapType(5, StringType(), 6, LongType(), value_required=False),
                required=False,
            )
        )
        row = {"s": {"m": [("k", 1)]}, "t": 1}
    else:
        field_type = StructType(NestedField(2, "x", LongType(), required=False))
        row = {"s": {"x": 1}, "t": 1}
    tbl = iceberg_catalog.create_table(
        "ns.unsupported",
        schema=pyiceberg.schema.Schema(
            NestedField(1, "s", field_type, required=False),
            NestedField(4, "t", LongType(), required=False),
        ),
    )
    if column == "nested_partition":
        with tbl.update_spec() as update:
            update.add_identity("s.x")
    tbl.append(pa.Table.from_pylist([row], schema=tbl.schema().as_arrow()))
    q = pl.scan_iceberg(tbl.metadata_location).select(projection)
    with pytest.raises(
        pl.exceptions.ComputeError, match=f"NotImplementedError: {re.escape(match)}"
    ):
        q.collect(engine=in_memory_engine)


def _partition_error_iceberg(
    iceberg_catalog: SqlCatalog, *, file_without_field: bool
) -> str:
    tbl = iceberg_catalog.create_table(
        "ns.partition_error",
        schema=pyiceberg.schema.Schema(NestedField(1, "a", LongType(), required=False)),
    )
    if file_without_field:
        tbl.append(pa.Table.from_pylist([{"a": 1}], schema=tbl.schema().as_arrow()))
    with tbl.update_schema() as update:
        update.add_column("d", DecimalType(9, 2))
    with tbl.update_spec() as update:
        update.add_identity("d")
    with tbl.update_schema() as update:
        update.update_column("d", field_type=DecimalType(18, 2))
    tbl.append(
        pa.Table.from_pylist(
            [{"a": 2, "d": Decimal("1.50")}, {"a": 3, "d": Decimal("2.50")}],
            schema=tbl.schema().as_arrow(),
        )
    )
    return tbl.metadata_location


def test_scan_iceberg_partition_error_reads_files(
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    location = _partition_error_iceberg(iceberg_catalog, file_without_field=False)
    assert_gpu_result_equal(
        pl.scan_iceberg(location), engine=engine, check_row_order=False
    )


def test_scan_iceberg_partition_error_missing_field(
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    q = pl.scan_iceberg(
        _partition_error_iceberg(iceberg_catalog, file_without_field=True)
    )
    match = "unsupported type change"
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(pl.exceptions.ComputeError, match=match),
            flatten_subgroups=True,
        ):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            q.collect(engine=engine)


def test_scan_iceberg_partition_error_sliced(
    in_memory_engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    q = pl.scan_iceberg(
        _partition_error_iceberg(iceberg_catalog, file_without_field=True)
    ).head(1)
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=re.escape(
            "NotImplementedError: "
            "Sliced Iceberg scan with partition values polars cannot supply"
        ),
    ):
        q.collect(engine=in_memory_engine)


def _write_with_field_ids(path: Path, table: pa.Table) -> None:
    fields = [
        pa.field(
            field.name, field.type, metadata={b"PARQUET:field_id": str(i).encode()}
        )
        for i, field in enumerate(table.schema, start=1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.cast(pa.schema(fields)), path)


def _scan_iceberg_files(
    paths: list[str], mapping: pa.Schema, **kwargs: Any
) -> pl.LazyFrame:
    # pyiceberg cannot write V3 tables or position deletes, so build the scan
    # that pl.scan_iceberg builds for such a table.
    return pl.scan_parquet(
        paths,
        schema=pl.DataFrame(mapping.empty_table()).schema,
        missing_columns="insert",
        extra_columns="ignore",
        cast_options=pl.ScanCastOptions._default_iceberg(),
        _column_mapping=("iceberg-column-mapping", mapping),
        **kwargs,
    )


@pytest.mark.parametrize(
    "query",
    [
        lambda q: q,
        lambda q: q.filter(pl.col("d") > 5),
        lambda q: q.select("d"),
    ],
    ids=["all", "filter", "select"],
)
def test_scan_iceberg_initial_defaults(
    engine: pl.GPUEngine, tmp_path: Path, query: Any
) -> None:
    paths = [tmp_path / "0.parquet", tmp_path / "1.parquet"]
    _write_with_field_ids(paths[0], pa.table({"a": pa.array([1, 2], pa.int64())}))
    _write_with_field_ids(
        paths[1],
        pa.table({"a": pa.array([3], pa.int64()), "d": pa.array([30], pa.int64())}),
    )
    q = _scan_iceberg_files(
        [str(path) for path in paths],
        pq.read_schema(paths[1]),
        _default_values=("iceberg", ({}, {2: pl.Series([7], dtype=pl.Int64)})),
    )
    assert_gpu_result_equal(query(q), engine=engine, check_row_order=False)


def test_scan_iceberg_nested_initial_default_unsupported(
    in_memory_engine: pl.GPUEngine, tmp_path: Path
) -> None:
    path = tmp_path / "0.parquet"
    mapping = pa.schema(
        [
            pa.field(
                "s",
                pa.struct(
                    [pa.field("x", pa.int64(), metadata={b"PARQUET:field_id": b"2"})]
                ),
                metadata={b"PARQUET:field_id": b"1"},
            )
        ]
    )
    pq.write_table(mapping.empty_table(), path)
    q = _scan_iceberg_files(
        [str(path)],
        mapping,
        _default_values=("iceberg", ({}, {2: pl.Series([1])})),
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def _puffin(deleted: dict[str, list[int]]) -> bytes:
    # pyroaring a dependency of pyiceberg
    # Could be removed once https://github.com/apache/iceberg-python/issues/1551
    # is addressed.
    pyroaring = pytest.importorskip("pyroaring")
    magic = b"PFA1"
    body = bytearray(magic)
    blobs = []
    for data_file, positions in deleted.items():
        payload = (
            bytes.fromhex("d1d33964")
            + (1).to_bytes(8, "little")
            + (0).to_bytes(4, "little")
            + pyroaring.BitMap(positions).serialize()
        )
        blob = len(payload).to_bytes(4, "big") + payload + bytes(4)
        blobs.append(
            {
                "type": "deletion-vector-v1",
                "fields": [],
                "snapshot-id": -1,
                "sequence-number": -1,
                "offset": len(body),
                "length": len(blob),
                "properties": {"referenced-data-file": data_file},
            }
        )
        body += blob
    footer = json.dumps({"blobs": blobs}).encode()
    return bytes(
        body + magic + footer + len(footer).to_bytes(4, "little") + bytes(4) + magic
    )


@pytest.fixture(
    params=["position_deletes", "deletion_vectors", "delta", "delta_partitioned"]
)
def deleted_rows_scan(request: pytest.FixtureRequest, tmp_path: Path) -> pl.LazyFrame:
    paths = []
    for index in range(3):
        path = tmp_path / "data" / f"part={index}" / "data.parquet"
        _write_with_field_ids(
            path,
            pa.table({"a": pa.array([4 * index + i for i in range(4)], pa.int64())}),
        )
        paths.append(str(path))
    deleted = {0: [0, 3], 2: [1]}
    if request.param == "position_deletes":
        position_deletes = {}
        for index, positions in deleted.items():
            path = tmp_path / "deletes" / f"{index}.parquet"
            pl.LazyFrame({"file_path": [paths[index]] * len(positions)}).with_columns(
                pos=pl.Series(positions, dtype=pl.Int64)
            ).sink_parquet(path, mkdir=True)
            position_deletes[index] = [str(path)]
        return _scan_iceberg_files(
            paths,
            pq.read_schema(paths[0]),
            _deletion_files=("iceberg", (position_deletes, {})),
        )
    if request.param == "deletion_vectors":
        puffin = tmp_path / "deletes.puffin"
        puffin.write_bytes(
            _puffin({paths[index]: positions for index, positions in deleted.items()})
        )
        return _scan_iceberg_files(
            paths,
            pq.read_schema(paths[0]),
            _deletion_files=("iceberg", ({}, dict.fromkeys(deleted, str(puffin)))),
        )
    selections = {
        paths[index]: [i not in positions for i in range(max(positions) + 1)]
        for index, positions in deleted.items()
    }

    def selection_vectors(requested: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(
            {"selection_vector": [selections.get(p) for p in requested["path"]]},
            schema={"selection_vector": pl.List(pl.Boolean)},
        )

    partitioned = request.param == "delta_partitioned"
    # deltalake cannot write deletion vectors, so build the scan that
    # pl.scan_delta builds for such a table.
    return pl.scan_parquet(
        paths,
        missing_columns="insert",
        extra_columns="ignore",
        cast_options=pl.ScanCastOptions._default_iceberg(),
        hive_partitioning=partitioned,
        hive_schema={"part": pl.Int64} if partitioned else None,
        _deletion_files=("delta-deletion-vector", selection_vectors),
    )


@pytest.mark.parametrize(
    "query",
    [
        lambda q: q.sort("a"),
        lambda q: q.with_row_index("idx"),
        lambda q: q.select(pl.len()),
    ],
    ids=["all", "row_index", "count"],
)
def test_scan_lake_deletions(
    engine: pl.GPUEngine, deleted_rows_scan: pl.LazyFrame, query: Any
) -> None:
    assert_gpu_result_equal(query(deleted_rows_scan), engine=engine)

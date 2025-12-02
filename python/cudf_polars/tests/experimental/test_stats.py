# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import pickle

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.io import _clear_source_info_cache
from cudf_polars.experimental.statistics import (
    apply_pkfk_heuristics,
    collect_base_stats,
    collect_statistics,
    find_equivalence_sets,
)
from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)
from cudf_polars.testing.io import make_lazy_frame, make_partitioned_source
from cudf_polars.utils.config import ConfigOptions


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "x": range(3_000),
            "y": ["cat", "dog", "fish"] * 1_000,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 600,
        }
    )


@pytest.fixture(scope="module")
def engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "shuffle_method": DEFAULT_RUNTIME,  # Names coincide
            "target_partition_size": 10_000,
            "max_rows_per_partition": 1_000,
            "stats_planning": {"use_reduction_planning": True},
        },
    )


def test_base_stats_dataframescan(df, engine):
    row_count = df.height
    q = pl.LazyFrame(df)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    stats = collect_base_stats(ir, ConfigOptions.from_polars_engine(engine))
    column_stats = stats.column_stats[ir]

    # Source info is the same for all columns
    source_info_x = column_stats["x"].source_info
    source_info_y = column_stats["y"].source_info
    source_info_z = column_stats["z"].source_info
    table_source_info = source_info_x.table_source_pairs[0].table_source
    assert table_source_info is source_info_y.table_source_pairs[0].table_source
    assert table_source_info is source_info_z.table_source_pairs[0].table_source
    assert source_info_x.row_count.value == row_count
    assert source_info_x.row_count.exact

    # Storage stats should not be available
    assert source_info_x.storage_size.value is None

    # Check unique stats.
    # We need to use force=True to sample unique-value statistics,
    # because nothing in the query requires unique-value statistics.
    assert math.isclose(
        source_info_x.unique_stats(force=True).count.value,
        row_count,
        rel_tol=5e-2,
    )
    assert math.isclose(
        source_info_x.unique_stats(force=True).fraction.value,
        1.0,
        abs_tol=1e-2,
    )
    assert not source_info_x.unique_stats(force=True).count.exact
    assert math.isclose(
        source_info_y.unique_stats(force=True).count.value,
        3,
        rel_tol=5e-2,
    )
    assert math.isclose(
        source_info_y.unique_stats(force=True).fraction.value,
        3 / row_count,
        abs_tol=1e-2,
    )
    assert not source_info_y.unique_stats(force=True).count.exact
    assert math.isclose(
        source_info_z.unique_stats(force=True).count.value,
        5,
        rel_tol=5e-2,
    )
    assert math.isclose(
        source_info_z.unique_stats(force=True).fraction.value,
        5 / row_count,
        abs_tol=1e-2,
    )
    assert not source_info_z.unique_stats(force=True).count.exact


@pytest.mark.parametrize("n_files", [1, 3])
@pytest.mark.parametrize("row_group_size", [None, 10_000])
@pytest.mark.parametrize("max_footer_samples", [3, 0])
@pytest.mark.parametrize("max_row_group_samples", [1, 0])
def test_base_stats_parquet(
    tmp_path,
    df,
    n_files,
    row_group_size,
    max_footer_samples,
    max_row_group_samples,
):
    _clear_source_info_cache()
    make_partitioned_source(
        df,
        tmp_path,
        "parquet",
        n_files=n_files,
        row_group_size=row_group_size,
    )
    q = pl.scan_parquet(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": 10_000,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
        parquet_options={
            "max_footer_samples": max_footer_samples,
            "max_row_group_samples": max_row_group_samples,
        },
    )
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    stats = collect_base_stats(ir, ConfigOptions.from_polars_engine(engine))
    column_stats = stats.column_stats[ir]

    # Source info is the same for all columns
    source_info_x = column_stats["x"].source_info
    source_info_y = column_stats["y"].source_info
    source_info_z = column_stats["z"].source_info
    table_source_info = source_info_x.table_source_pairs[0].table_source
    assert table_source_info is source_info_y.table_source_pairs[0].table_source
    assert table_source_info is source_info_z.table_source_pairs[0].table_source
    if max_footer_samples:
        assert source_info_x.row_count.value == df.height
        assert source_info_x.row_count.exact
    else:
        assert source_info_x.row_count.value is None

    # Storage stats should be available
    if max_footer_samples:
        assert source_info_x.storage_size.value > 0
        assert source_info_y.storage_size.value > 0
    else:
        assert source_info_x.storage_size.value is None
        assert source_info_y.storage_size.value is None

    # All read columns should be marked
    assert set(table_source_info._read_columns) == {"x", "y", "z"}

    # source._unique_stats should be empty
    assert set(table_source_info._unique_stats) == set()

    if max_footer_samples and max_row_group_samples:
        assert source_info_x.unique_stats(force=True).count.value == df.height
        assert source_info_x.unique_stats(force=True).fraction.value == 1.0
    else:
        assert source_info_x.unique_stats(force=True).count.value is None

    # source_info._unique_stats should only contain 'x'
    if max_footer_samples and max_row_group_samples:
        assert set(table_source_info._unique_stats) == {"x"}
    else:
        assert set(table_source_info._unique_stats) == set()

    # Check add_unique_stats_column behavior
    if max_footer_samples and max_row_group_samples:
        # Can add a "bad"/missing key column
        source_info_x.add_unique_stats_column("foo")
        assert set(table_source_info._unique_stats) == {"x"}

        # Mark 'z' as a key column, and query 'y' stats
        source_info_z.add_unique_stats_column()
        if n_files == 1 and row_group_size == 10_000:
            assert source_info_y.unique_stats(force=True).count.value == 3
        else:
            assert source_info_y.unique_stats(force=True).count.value is None
        assert source_info_y.unique_stats(force=True).fraction.value < 1.0

        # source_info._unique_stats should contain all columns now
        assert set(table_source_info._unique_stats) == {"x", "y", "z"}


def test_base_stats_csv(engine, tmp_path, df):
    q = make_lazy_frame(df, "csv", path=tmp_path, n_files=3)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    stats = collect_base_stats(ir, ConfigOptions.from_polars_engine(engine))
    column_stats = stats.column_stats[ir]

    # Source info should be empty for CSV
    source_info_x = column_stats["x"].source_info
    assert source_info_x.row_count.value is None
    assert source_info_x.unique_stats().count.value is None
    assert source_info_x.unique_stats().fraction.value is None


@pytest.mark.parametrize("max_footer_samples", [1, 3])
@pytest.mark.parametrize("max_row_group_samples", [1, 2])
def test_base_stats_parquet_groupby(
    tmp_path,
    df,
    max_footer_samples,
    max_row_group_samples,
):
    n_files = 3
    _clear_source_info_cache()
    make_partitioned_source(df, tmp_path, "parquet", n_files=n_files)
    q = pl.scan_parquet(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": 10_000,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "stats_planning": {"use_reduction_planning": True},
        },
        parquet_options={
            "max_footer_samples": max_footer_samples,
            "max_row_group_samples": max_row_group_samples,
        },
    )

    # Check simple selection
    q1 = q.select(pl.col("x"), pl.col("y"))
    qir1 = Translator(q1._ldf.visit(), engine).translate_ir()
    stats = collect_base_stats(qir1, ConfigOptions.from_polars_engine(engine))
    source_info_y = stats.column_stats[qir1]["y"].source_info
    unique_stats_y = source_info_y.unique_stats(force=True)
    y_unique_fraction = unique_stats_y.fraction
    y_row_count = source_info_y.row_count
    assert y_unique_fraction.value < 1.0
    assert y_unique_fraction.value > 0.0
    assert unique_stats_y.count.value is None
    if max_footer_samples >= n_files:
        # We should have "exact" row-count statistics
        assert y_row_count.value == df.height
        assert y_row_count.exact
    else:
        # We should have "estimated" row-count statistics
        assert y_row_count.value > 0
        assert not y_row_count.exact
    assert_gpu_result_equal(q1.sort(pl.col("x")).slice(0, 2), engine=engine)

    # Source statistics of "y" should match after GroupBy/Select/HStack/etc
    q2 = (
        pl.concat(
            [
                q.select(pl.col("x")),
                q.select(pl.col("y")),
            ],
            how="horizontal",
        )
        .group_by(pl.col("y"))
        .sum()
        .select(pl.col("x").max(), pl.col("y"))
        .with_columns((pl.col("x") * pl.col("x")).alias("x2"))
    )
    qir2 = Translator(q2._ldf.visit(), engine).translate_ir()
    stats = collect_base_stats(qir2, ConfigOptions.from_polars_engine(engine))
    source_info_y = stats.column_stats[qir2]["y"].source_info
    assert source_info_y.unique_stats().fraction == y_unique_fraction
    assert y_row_count == source_info_y.row_count
    assert_gpu_result_equal(q2.sort(pl.col("y")).slice(0, 2), engine=engine)


@pytest.mark.parametrize("how", ["inner", "left", "right"])
def test_base_stats_join(how, engine):
    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(9),
            "y": [2, 4, 3] * 3,
            "z": [1, 2, 3] * 3,
        }
    )
    q = left.join(right, on="y", how=how)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    stats = collect_base_stats(ir, ConfigOptions.from_polars_engine(engine))

    ir_column_stats = stats.column_stats[ir]
    left_count, right_count = 15, 9
    if how == "left":
        assert ir_column_stats["x"].source_info.row_count.value == left_count
        assert ir_column_stats["y"].source_info.row_count.value == left_count
        assert ir_column_stats["z"].source_info.row_count.value == left_count
    if how == "inner":
        assert ir_column_stats["x"].source_info.row_count.value == left_count
        assert ir_column_stats["y"].source_info.row_count.value == left_count
        assert ir_column_stats["z"].source_info.row_count.value == left_count
        assert ir_column_stats["xx"].source_info.row_count.value == right_count
        assert ir_column_stats["z_right"].source_info.row_count.value == right_count
    if how == "right":
        assert ir_column_stats["xx"].source_info.row_count.value == right_count
        assert ir_column_stats["y"].source_info.row_count.value == right_count
        assert ir_column_stats["z"].source_info.row_count.value == right_count


def test_base_stats_union(engine):
    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "x": range(9),
            "y": [2, 4, 3] * 3,
            "z": [1.0, 2.0, 3.0] * 3,
        }
    )

    q = pl.concat([left, right])
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    stats = collect_base_stats(ir, ConfigOptions.from_polars_engine(engine))
    column_stats = stats.column_stats[ir]

    # Source row-count estimate is the sum of the two sources
    source_info_x = column_stats["x"].source_info
    assert source_info_x.row_count.value == 24


def test_base_stats_distinct(engine, df):
    row_count = df.height
    q = pl.LazyFrame(df).unique(subset=["y"])
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    stats = collect_base_stats(ir, ConfigOptions.from_polars_engine(engine))
    column_stats = stats.column_stats[ir]

    source_info_y = column_stats["y"].source_info
    assert source_info_y.row_count.value == row_count
    assert source_info_y.row_count.exact


def test_base_stats_join_key_info(engine):
    # Customers table (PK: cust_id)
    customers = pl.LazyFrame(
        {
            "cust_id": [1, 2],
            "cust_name": ["Alice", "Bob"],
        }
    )

    # Orders table (PK: order_id)
    orders = pl.LazyFrame(
        {
            "order_id": [100, 101, 102],
            "cust_id": [1, 2, 1],
            "prod_id": [10, 20, 10],
            "loc_id": [501, 501, 502],
            "quant": [2, 1, 4],
        }
    )

    # locations table (PK: prod_id, loc_id)
    locations = pl.LazyFrame(
        {
            "prod_id": [10, 20, 10],
            "loc_id": [501, 501, 502],
            "price": [50, 60, 55],
        }
    )

    # Step 1: Multi-key join orders and locations on prod_id & loc_id
    orders_with_price = orders.join(locations, on=["prod_id", "loc_id"], how="inner")

    # Step 2: Join result to customers on cust_id
    q = orders_with_price.join(customers, on="cust_id", how="inner")

    ir = Translator(q._ldf.visit(), engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(engine)
    stats = collect_base_stats(ir, config_options)
    join_info = stats.join_info

    # Check equivalence sets
    key_sets = sorted(
        sorted(tuple(cs.name for cs in k.column_stats) for k in group)
        for group in find_equivalence_sets(join_info.key_map)
    )
    assert len(key_sets) == 2
    assert key_sets[0] == [("cust_id",), ("cust_id",)]
    assert key_sets[1] == [("prod_id", "loc_id"), ("prod_id", "loc_id")]

    # Check basic PK-FK unique-count heuristics
    apply_pkfk_heuristics(join_info)
    implied_unique_count = join_info.join_map[ir][0].implied_unique_count
    assert implied_unique_count == join_info.join_map[ir][1].implied_unique_count
    assert (
        q.select(pl.col("cust_id").n_unique()).collect().item() == implied_unique_count
    )
    assert (
        # Calling apply_pkfk_heuristics should update the implied_unique_count
        # estimate on the associated ColumnSourceInfo as well
        stats.column_stats[ir]["cust_id"].source_info.implied_unique_count.value
        == implied_unique_count
    )

    # Check basic collect_statistics behavior
    stats = collect_statistics(ir, config_options)
    local_unique_count = stats.column_stats[ir]["cust_id"].unique_count.value
    source_unique_count = (
        stats.column_stats[ir]["cust_id"].source_info.unique_stats().count.value
    )
    assert local_unique_count == source_unique_count
    assert stats.row_count[ir].value == q.collect().height


def test_dataframescan_stats_pickle(engine):
    df = pl.DataFrame({"x": range(100), "y": [1, 2] * 50})
    q = pl.LazyFrame(df)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    stats = collect_base_stats(ir, ConfigOptions.from_polars_engine(engine))

    # Pickle and unpickle the stats collector
    pickled = pickle.dumps(stats)
    unpickled_stats = pickle.loads(pickled)

    # Verify the unpickled stats are equivalent
    assert type(unpickled_stats) is type(stats)
    assert unpickled_stats.column_stats[ir]["x"].source_info.row_count.value == 100


@pytest.mark.parametrize("use_io_partitioning", [True, False])
@pytest.mark.parametrize("use_reduction_planning", [True, False])
@pytest.mark.parametrize("use_join_heuristics", [True, False])
@pytest.mark.parametrize("use_sampling", [True, False])
@pytest.mark.parametrize("default_selectivity", [0.5, 1.0])
@pytest.mark.parametrize("kind", ["parquet", "csv", "frame"])
def test_stats_planning(
    tmp_path,
    kind,
    use_io_partitioning,
    use_reduction_planning,
    use_join_heuristics,
    use_sampling,
    default_selectivity,
):
    # Create temporary GPU Engine
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "shuffle_method": DEFAULT_RUNTIME,  # Names coincide
            "target_partition_size": 10_000,
            "max_rows_per_partition": 1_000,
            "stats_planning": {
                "use_io_partitioning": use_io_partitioning,
                "use_reduction_planning": use_reduction_planning,
                "use_join_heuristics": use_join_heuristics,
                "use_sampling": use_sampling,
                "default_selectivity": default_selectivity,
            },
        },
    )

    # Define "complicated" query that uses all stats planning options
    sales = pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "customer_id": [101, 102, 101, 103, 102, 101],
            "amount": [50.0, 75.0, 30.0, 120.0, 85.0, 40.0],
            "product": ["A", "B", "A", "C", "B", "A"],
        }
    )
    sales = make_lazy_frame(sales, kind, path=tmp_path / f"sales_{kind}")
    customers = pl.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "customer_name": ["Alice", "Bob", "Charlie"],
            "region": ["North", "South", "North"],
        }
    )
    customers = make_lazy_frame(customers, kind, path=tmp_path / f"customers_{kind}")
    q_join = sales.filter(pl.col("amount") < 100.0).join(
        customers, on="customer_id", how="inner"
    )
    q_gb = q_join.group_by("customer_id").agg(
        [
            pl.col("amount").sum().alias("total_amount"),
            pl.col("order_id").count().alias("order_count"),
            pl.col("customer_name").first().alias("name"),
            pl.col("region").first().alias("region"),
        ]
    )
    q = q_gb.sort("customer_id")

    # Verify the query runs correctly
    assert_gpu_result_equal(q, engine=engine)

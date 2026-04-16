# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.experimental.explain import explain_query, serialize_query
from cudf_polars.testing.asserts import DEFAULT_CLUSTER, DEFAULT_RUNTIME
from cudf_polars.testing.io import make_partitioned_source


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "x": range(25_000),
            "y": ["cat", "dog"] * 12_500,
            "z": [1.0, 2.0] * 12_500,
        }
    )


def test_explain_physical_plan(tmp_path, df):
    make_partitioned_source(df, tmp_path, fmt="parquet", n_files=5)

    q = (
        pl.scan_parquet(tmp_path)
        .filter((pl.col("x") < 40_000) & (pl.col("z") > 1.0))
        .with_columns((pl.col("x") + pl.col("z")).alias("sum"))
        .select(["sum", "y"])
    )

    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
        executor_options={
            "target_partition_size": 10_000,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )

    plan = explain_query(q, engine)

    if DEFAULT_RUNTIME == "tasks":
        # rapidsmpf runtime does not split Scan nodes at lowering time
        assert "UNION" in plan
        assert "SPLITSCAN" in plan
    assert "SELECT ('sum', 'y')" in plan or "PROJECTION ('sum', 'y')" in plan


def test_shuffle_properties():
    # Join with broadcast_join_limit=1 forces shuffle-based join, producing
    # Shuffle nodes in the lowered plan.
    left = pl.LazyFrame({"a": ["x", "y", "x"], "b": [1, 2, 3]})
    right = pl.LazyFrame({"a": ["x", "y", "z"], "c": [4, 5, 6]})
    q = left.join(right, on="a", how="inner")
    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
        executor_options={
            "max_rows_per_partition": 1,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "shuffle_method": DEFAULT_RUNTIME,
            "broadcast_join_limit": 1,
            "dynamic_planning": None,  # Requires static planning
        },
    )
    dag = serialize_query(q, engine)

    shuffle_nodes = [n for n in dag.nodes.values() if n.type == "Shuffle"]
    assert len(shuffle_nodes) >= 1, "Expected at least one Shuffle node in lowered plan"
    node = shuffle_nodes[0]

    if DEFAULT_RUNTIME == "tasks":
        shuffle_method = "tasks"
    elif DEFAULT_CLUSTER == "single":
        shuffle_method = "rapidsmpf-single"
    else:
        shuffle_method = "rapidsmpf"

    assert node.properties == {
        "keys": ["a"],
        "shuffle_method": shuffle_method,
    }

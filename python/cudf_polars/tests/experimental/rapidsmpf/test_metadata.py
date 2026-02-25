# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RapidsMPF metadata functionality."""

from __future__ import annotations

import pytest
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)

import polars as pl

from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import Select
from cudf_polars.experimental.rapidsmpf.core import evaluate_logical_plan
from cudf_polars.experimental.rapidsmpf.utils import (
    get_partitioning_moduli,
    remap_partitioning,
    remap_partitioning_select,
)
from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
)
from cudf_polars.utils.config import ConfigOptions


@pytest.fixture(scope="module")
def left() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )


@pytest.fixture(scope="module")
def right() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "xx": range(6),
            "y": [2, 4, 3] * 2,
            "zz": [1, 2] * 3,
        }
    )


@pytest.mark.skipif(
    DEFAULT_RUNTIME != "rapidsmpf", reason="Requires 'rapidsmpf' runtime."
)
@pytest.mark.skipif(DEFAULT_CLUSTER != "single", reason="Requires 'single' cluster.")
@pytest.mark.parametrize("broadcast_join_limit", [2, 10])
def test_rapidsmpf_join_metadata(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    broadcast_join_limit: int,
) -> None:
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 1,
            "broadcast_join_limit": broadcast_join_limit,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )
    config_options = ConfigOptions.from_polars_engine(engine)
    q = left.join(
        right,
        on="y",
        how="left",
    ).filter(pl.col("x") > pl.col("zz"))
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    left_count = left.collect().height
    right_count = right.collect().height

    metadata_collector = evaluate_logical_plan(
        ir, config_options, collect_metadata=True
    )[1]
    assert metadata_collector is not None
    assert len(metadata_collector) == 1
    metadata = metadata_collector[0]
    assert metadata.local_count == left_count
    assert metadata.duplicated is False
    if right_count > broadcast_join_limit:
        # After shuffle, partitioning has inter_rank=HashScheme, local="inherit"
        assert isinstance(metadata.partitioning.inter_rank, HashScheme)
        # "y" is at index 1 in the output schema: ["x", "y", "z", "xx", "zz"]
        assert metadata.partitioning.inter_rank.column_indices == (1,)
        assert metadata.partitioning.local == "inherit"
    else:
        # No partitioning (broadcast join preserves no partitioning from IO)
        assert metadata.partitioning.inter_rank is None
        assert metadata.partitioning.local is None


@pytest.mark.parametrize(
    "local_count,partitioning,key_indices,nranks,expected",
    [
        (4, None, (0, 1), 1, (1, 0)),
        (4, None, (0, 1), 4, (0, 0)),
        (
            8,
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0, 1),
            4,
            (8, None),
        ),
        (
            4,
            Partitioning(
                inter_rank=HashScheme((0, 1), 8),
                local=HashScheme((0, 1), 4),
            ),
            (0, 1),
            4,
            (8, 4),
        ),
        (
            8,
            Partitioning(
                inter_rank=HashScheme((0, 1), 8),
                local=HashScheme((0,), 4),
            ),
            (0, 1),
            4,
            (8, 0),
        ),
        (
            8,  # local_count != local modulus
            Partitioning(
                inter_rank=HashScheme((0, 1), 8),
                local=HashScheme((0, 1), 4),
            ),
            (0, 1),
            4,
            (8, 0),
        ),
        (
            8,
            Partitioning(inter_rank=HashScheme((0,), 8), local="inherit"),
            (0, 1),
            4,
            (0, 0),
        ),
        (
            8,
            Partitioning(inter_rank=HashScheme((1, 0), 8), local="inherit"),
            (0, 1),
            4,
            (0, 0),
        ),
        (
            4,
            Partitioning(
                inter_rank=None,
                local=HashScheme((0, 1), 4),
            ),
            (0, 1),
            1,
            (4, None),
        ),
        (
            4,
            Partitioning(
                inter_rank=None,
                local=HashScheme((0, 1), 4),
            ),
            (0, 1),
            4,
            (0, 0),
        ),
        (
            8,
            Partitioning(
                inter_rank=HashScheme((0, 1), 8),
                local=None,
            ),
            (0, 1),
            4,
            (8, 0),
        ),
    ],
)
def test_get_partitioning_moduli(
    local_count, partitioning, key_indices, nranks, expected
) -> None:
    """get_partitioning_moduli returns (inter_rank_modulus, local_modulus)."""
    metadata = ChannelMetadata(
        local_count=local_count,
        partitioning=partitioning,
    )
    assert get_partitioning_moduli(metadata, key_indices, nranks) == expected


@pytest.mark.parametrize(
    "local_count,partitioning,key_indices,nranks,expected",
    [
        # Partitioned on (0,); keys (0, 1) → prefix (0,) matches
        (
            8,
            Partitioning(inter_rank=HashScheme((0,), 8), local="inherit"),
            (0, 1),
            4,
            (8, None),
        ),
        # Partitioned on (0, 1); keys (0, 1, 2) → prefix (0, 1) matches
        (
            8,
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0, 1, 2),
            4,
            (8, None),
        ),
        # Partitioned on (0,) with explicit local; keys (0, 1) → prefix matches
        (
            4,
            Partitioning(
                inter_rank=HashScheme((0,), 8),
                local=HashScheme((0,), 4),
            ),
            (0, 1),
            4,
            (8, 4),
        ),
        # Full key match with allow_subset: same as exact match
        (
            8,
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0, 1),
            4,
            (8, None),
        ),
        # Keys (0,) are shorter than partition (0, 1) → no prefix match
        (
            8,
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0,),
            4,
            (0, 0),
        ),
        # Partitioned on (1,); keys (0, 1) → prefix of keys is (0,), not (1,) → no match
        (
            8,
            Partitioning(inter_rank=HashScheme((1,), 8), local="inherit"),
            (0, 1),
            4,
            (0, 0),
        ),
    ],
)
def test_get_partitioning_moduli_allow_subset(
    local_count, partitioning, key_indices, nranks, expected
) -> None:
    """get_partitioning_moduli with allow_subset=True matches on prefix of key_indices."""
    metadata = ChannelMetadata(
        local_count=local_count,
        partitioning=partitioning,
    )
    assert (
        get_partitioning_moduli(metadata, key_indices, nranks, allow_subset=True)
        == expected
    )


def _make_select_ir(engine: pl.GPUEngine, output_columns: tuple[str, ...]):
    q = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    child = Translator(q._ldf.visit(), engine).translate_ir()
    out_schema = {k: child.schema[k] for k in output_columns}
    exprs = tuple(
        expr.NamedExpr(name, expr.Col(child.schema[name], name))
        for name in output_columns
    )
    return Select(out_schema, exprs, should_broadcast=False, df=child)


def test_remap_partitioning_select_none_input() -> None:
    engine = pl.GPUEngine(executor="streaming")
    assert remap_partitioning_select(_make_select_ir(engine, ("a", "b")), None) is None


def test_remap_partitioning_select_preserves_keys() -> None:
    engine = pl.GPUEngine(executor="streaming")
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = remap_partitioning_select(_make_select_ir(engine, ("a", "b")), part)
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (0, 1)
    assert result.inter_rank.modulus == 8
    assert result.local == "inherit"


def test_remap_partitioning_select_drops_key() -> None:
    engine = pl.GPUEngine(executor="streaming")
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = remap_partitioning_select(_make_select_ir(engine, ("a",)), part)
    assert result is not None
    assert result.inter_rank is None
    assert result.local == "inherit"


def test_remap_partitioning_select_renamed_key() -> None:
    """Partitioning is preserved when a key column is renamed in the Select."""
    engine = pl.GPUEngine(executor="streaming")
    q = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    child = Translator(q._ldf.visit(), engine).translate_ir()
    # Output (a_renamed, b) where a_renamed is Col("a")
    out_schema = {"a_renamed": child.schema["a"], "b": child.schema["b"]}
    exprs = (
        expr.NamedExpr("a_renamed", expr.Col(child.schema["a"], "a")),
        expr.NamedExpr("b", expr.Col(child.schema["b"], "b")),
    )
    select = Select(out_schema, exprs, should_broadcast=False, df=child)
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = remap_partitioning_select(select, part)
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (0, 1)  # a_renamed, b in output
    assert result.inter_rank.modulus == 8
    assert result.local == "inherit"


def test_remap_partitioning_reorder_columns() -> None:
    old_schema = {
        "a": DataType(pl.Int64),
        "b": DataType(pl.Int64),
        "c": DataType(pl.Int64),
    }
    new_schema = {"b": DataType(pl.Int64), "a": DataType(pl.Int64)}
    result = remap_partitioning(
        Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
        old_schema,
        new_schema,
    )
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (1, 0)
    assert result.inter_rank.modulus == 8

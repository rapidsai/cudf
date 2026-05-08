# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RapidsMPF metadata functionality."""

from __future__ import annotations

import pytest
from rapidsmpf.streaming.cudf.channel_metadata import (
    HashScheme,
    Partitioning,
)

import polars as pl

from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import GroupBy, HStack, Projection, Select
from cudf_polars.experimental.rapidsmpf.core import evaluate_logical_plan
from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.experimental.rapidsmpf.utils import (
    NormalizedPartitioning,
    maybe_remap_partitioning,
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


@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(
            max_rows_per_partition=1,
            broadcast_join_limit=2,
            dynamic_planning=None,
        ),
        StreamingOptions(
            max_rows_per_partition=1,
            broadcast_join_limit=10,
            dynamic_planning=None,
        ),
    ],
)
def test_rapidsmpf_join_metadata(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    spmd_engine_factory,
    options,
) -> None:
    # Pinned to SPMD: ``ChannelMetadata.__reduce_cython__`` can't pickle
    # ``self._handle`` across worker/actor processes, so the
    # ``metadata_collector`` round-trip fails on Dask and Ray.
    #
    # When https://github.com/rapidsai/cudf/pull/22394 lands, dedup of
    # replicated outputs moves to the Dask/Ray frontends and the
    # ``duplicated`` flag's semantics change to "every rank holds the
    # data". Revisit the ``len(metadata_collector) == 1`` and
    # ``metadata.duplicated is False`` assertions below, and reconsider
    # whether this test can widen to ``streaming_engine_factory``.
    engine = spmd_engine_factory(options)
    config_options = ConfigOptions.from_polars_engine(engine)
    broadcast_join_limit = config_options.executor.broadcast_join_limit
    q = left.join(
        right,
        on="y",
        how="left",
    ).filter(pl.col("x") > pl.col("zz"))
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    left_count = left.collect(engine=engine).height
    right_count = right.collect(engine=engine).height

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
    "partitioning,key_indices,nranks,expected",
    [
        (
            None,
            (0, 1),
            1,
            NormalizedPartitioning(HashScheme((0, 1), 1), None),
        ),
        (None, (0, 1), 4, NormalizedPartitioning(None, None)),
        (
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
        ),
        (
            Partitioning(
                inter_rank=HashScheme((0, 1), 8),
                local=HashScheme((0, 1), 4),
            ),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), HashScheme((0, 1), 4)),
        ),
        (
            Partitioning(
                inter_rank=HashScheme((0, 1), 8),
                local=HashScheme((0,), 4),
            ),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), None),
        ),
        (
            Partitioning(inter_rank=HashScheme((0,), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(None, None),
        ),
        (
            Partitioning(inter_rank=HashScheme((1, 0), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(None, None),
        ),
        (
            Partitioning(
                inter_rank=None,
                local=HashScheme((0, 1), 4),
            ),
            (0, 1),
            1,
            NormalizedPartitioning(HashScheme((0, 1), 4), "inherit"),
        ),
        (
            Partitioning(
                inter_rank=None,
                local=HashScheme((0, 1), 4),
            ),
            (0, 1),
            4,
            NormalizedPartitioning(None, None),
        ),
        (
            Partitioning(
                inter_rank=HashScheme((0, 1), 8),
                local=None,
            ),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), None),
        ),
    ],
)
def test_get_partitioning_moduli(partitioning, key_indices, nranks, expected) -> None:
    """from_keys(..., allow_subset=False) matches expected NormalizedPartitioning."""
    state = NormalizedPartitioning.from_keys(
        partitioning, nranks, indices=key_indices, allow_subset=False
    )
    assert state == expected


@pytest.mark.parametrize(
    "partitioning,key_indices,nranks,expected",
    [
        # Partitioned on (0,); keys (0, 1) → prefix (0,) matches
        (
            Partitioning(inter_rank=HashScheme((0,), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0,), 8), "inherit"),
        ),
        # Partitioned on (0, 1); keys (0, 1, 2) → prefix (0, 1) matches
        (
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0, 1, 2),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
        ),
        # Partitioned on (0,) with explicit local; keys (0, 1) → prefix matches
        (
            Partitioning(
                inter_rank=HashScheme((0,), 8),
                local=HashScheme((0,), 4),
            ),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0,), 8), HashScheme((0,), 4)),
        ),
        # Full key match with allow_subset: same as exact match
        (
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
        ),
        # Keys (0,) are shorter than partition (0, 1) → no prefix match
        (
            Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit"),
            (0,),
            4,
            NormalizedPartitioning(None, None),
        ),
        # Partitioned on (1,); keys (0, 1) → prefix of keys is (0,), not (1,) → no match
        (
            Partitioning(inter_rank=HashScheme((1,), 8), local="inherit"),
            (0, 1),
            4,
            NormalizedPartitioning(None, None),
        ),
        # Resolves https://github.com/rapidsai/cudf/issues/21742
        (
            Partitioning(inter_rank=HashScheme((0,), 8), local="inherit"),
            (1,),
            1,
            NormalizedPartitioning(HashScheme((1,), 1), None),
        ),
    ],
)
def test_get_partitioning_moduli_allow_subset(
    partitioning, key_indices, nranks, expected
) -> None:
    """from_keys(..., allow_subset=True) matches expected NormalizedPartitioning."""
    state = NormalizedPartitioning.from_keys(
        partitioning, nranks, indices=key_indices, allow_subset=True
    )
    assert state == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        # Same inter_rank modulus and key count, both "inherit" → aligned
        (
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            NormalizedPartitioning(HashScheme((2, 3), 8), "inherit"),
            True,
        ),
        # Same modulus, same local HashScheme arity → aligned (column_indices not compared)
        (
            NormalizedPartitioning(HashScheme((0,), 8), HashScheme((0,), 4)),
            NormalizedPartitioning(HashScheme((1,), 8), HashScheme((1,), 4)),
            True,
        ),
        # Different inter_rank modulus → not aligned
        (
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            NormalizedPartitioning(HashScheme((0, 1), 4), "inherit"),
            False,
        ),
        # Different key count → not aligned
        (
            NormalizedPartitioning(HashScheme((0,), 8), "inherit"),
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            False,
        ),
        # One side has local=None → not aligned (bool is False)
        (
            NormalizedPartitioning(HashScheme((0,), 8), "inherit"),
            NormalizedPartitioning(HashScheme((0,), 8), None),
            False,
        ),
        # Both None inter_rank → not aligned
        (
            NormalizedPartitioning(None, None),
            NormalizedPartitioning(None, None),
            False,
        ),
        # Mismatched local types: "inherit" vs HashScheme → not aligned
        (
            NormalizedPartitioning(HashScheme((0,), 8), "inherit"),
            NormalizedPartitioning(HashScheme((0,), 8), HashScheme((0,), 4)),
            False,
        ),
        # Local HashSchemes with different modulus → not aligned
        (
            NormalizedPartitioning(HashScheme((0,), 8), HashScheme((0,), 4)),
            NormalizedPartitioning(HashScheme((0,), 8), HashScheme((0,), 2)),
            False,
        ),
        # Inter-rank aligned; local HashSchemes same modulus but different arity → not aligned
        (
            NormalizedPartitioning(HashScheme((0, 1), 8), HashScheme((0,), 4)),
            NormalizedPartitioning(HashScheme((0, 1), 8), HashScheme((0, 1), 4)),
            False,
        ),
        # Reflexive when both sides fully partitioned and identical
        (
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            True,
        ),
        # Inter-rank missing on one side → not aligned
        (
            NormalizedPartitioning(None, "inherit"),
            NormalizedPartitioning(HashScheme((0, 1), 8), "inherit"),
            False,
        ),
    ],
)
def test_is_aligned_with(left, right, expected) -> None:
    """is_aligned_with checks compatible hash layout for chunkwise operations."""
    assert left.is_aligned_with(right) is expected
    assert right.is_aligned_with(left) is expected


def test_normalized_partitioning_eq() -> None:
    a = NormalizedPartitioning(HashScheme((0, 1), 8), "inherit")
    b = NormalizedPartitioning(HashScheme((0, 1), 8), "inherit")
    c = NormalizedPartitioning(HashScheme((0, 1), 4), "inherit")
    assert a == b
    assert a != c


def _make_select_ir(engine: pl.GPUEngine, output_columns: tuple[str, ...]):
    q = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    child = Translator(q._ldf.visit(), engine).translate_ir()
    out_schema = {k: child.schema[k] for k in output_columns}
    exprs = tuple(
        expr.NamedExpr(name, expr.Col(child.schema[name], name))
        for name in output_columns
    )
    return Select(out_schema, exprs, should_broadcast=False, df=child)


def test_remap_partitioning_select_none_input(streaming_engine) -> None:
    assert (
        maybe_remap_partitioning(_make_select_ir(streaming_engine, ("a", "b")), None)
        is None
    )


def test_remap_partitioning_select_preserves_keys(streaming_engine) -> None:
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(
        _make_select_ir(streaming_engine, ("a", "b")), part
    )
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (0, 1)
    assert result.inter_rank.modulus == 8
    assert result.local == "inherit"


def test_remap_partitioning_groupby(streaming_engine) -> None:
    """Hash indices refer to the groupby input child; remap to groupby output columns."""
    q = (
        pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
        .group_by("a", "b")
        .agg(pl.col("c").sum())
    )
    ir = Translator(q._ldf.visit(), streaming_engine).translate_ir()
    while isinstance(ir, (Select, Projection)):
        ir = ir.children[0]
    assert isinstance(ir, GroupBy)

    gb = ir
    key_names = tuple(ne.name for ne in gb.keys)
    child_cols = list(gb.children[0].schema.keys())
    input_indices = tuple(child_cols.index(n) for n in key_names)
    out_cols = list(gb.schema.keys())
    expected = tuple(out_cols.index(n) for n in key_names)

    part = Partitioning(inter_rank=HashScheme(input_indices, 8), local="inherit")
    result = maybe_remap_partitioning(gb, part)
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == expected
    assert result.inter_rank.modulus == 8
    assert result.local == "inherit"


def test_remap_partitioning_hstack_appends_preserves_keys(streaming_engine) -> None:
    q = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    child = Translator(q._ldf.visit(), streaming_engine).translate_ir()
    d_dtype = DataType(pl.Int64())
    hstack = HStack(
        {**child.schema, "d": d_dtype},
        (expr.NamedExpr("d", expr.Literal(d_dtype, 0)),),
        should_broadcast=True,
        df=child,
    )
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(hstack, part)
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (0, 1)
    assert result.inter_rank.modulus == 8
    assert result.local == "inherit"


def test_remap_partitioning_select_drops_key(streaming_engine) -> None:
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(_make_select_ir(streaming_engine, ("a",)), part)
    assert result is not None
    assert result.inter_rank is None
    assert result.local == "inherit"


def test_remap_partitioning_select_renamed_key(streaming_engine) -> None:
    q = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    child = Translator(q._ldf.visit(), streaming_engine).translate_ir()
    # Output (a_renamed, b) where a_renamed is Col("a")
    out_schema = {"a_renamed": child.schema["a"], "b": child.schema["b"]}
    exprs = (
        expr.NamedExpr("a_renamed", expr.Col(child.schema["a"], "a")),
        expr.NamedExpr("b", expr.Col(child.schema["b"], "b")),
    )
    select = Select(out_schema, exprs, should_broadcast=False, df=child)
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(select, part)
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (0, 1)  # a_renamed, b in output
    assert result.inter_rank.modulus == 8
    assert result.local == "inherit"


def test_remap_partitioning_reorder_columns(streaming_engine) -> None:
    # Select (b, a) from (a, b, c) -> partition keys (a,b) become indices (1, 0) in output
    select = _make_select_ir(streaming_engine, ("b", "a"))
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(select, part)
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (1, 0)
    assert result.inter_rank.modulus == 8


def test_remap_partitioning_reorder_columns_projection(streaming_engine) -> None:
    q = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    child = Translator(q._ldf.visit(), streaming_engine).translate_ir()
    # Projection output (b, a) -> child has (a, b, c); partition keys (a,b) -> indices (1, 0)
    out_schema = {k: child.schema[k] for k in ("b", "a")}
    proj = Projection(out_schema, child)
    part = Partitioning(inter_rank=HashScheme((0, 1), 8), local="inherit")
    result = maybe_remap_partitioning(proj, part, child_ir=proj.children[0])
    assert result is not None
    assert result.inter_rank is not None
    assert result.inter_rank.column_indices == (1, 0)
    assert result.inter_rank.modulus == 8

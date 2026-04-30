# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(fallback_mode="warn"),
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": range(150),
            "y": list(range(30)) * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 30,
        }
    )


@pytest.mark.filterwarnings("ignore:Unsupported unique options for multiple partitions")
@pytest.mark.parametrize("subset", [None, ("y",), ("y", "z")])
@pytest.mark.parametrize("keep", ["first", "last", "any", "none"])
@pytest.mark.parametrize("maintain_order", [True, False])
@pytest.mark.parametrize("cardinality", [{}, {"y": 0.7}])
def test_unique(
    df, streaming_engine_factory, keep, subset, maintain_order, cardinality
):
    engine = streaming_engine_factory(
        StreamingOptions(unique_fraction=cardinality, fallback_mode="warn"),
    )
    q = df.unique(subset=subset, keep=keep, maintain_order=maintain_order)
    check_row_order = maintain_order
    if keep == "any" and subset:
        q = q.select(*(pl.col(col) for col in subset))
        check_row_order = False

    assert_gpu_result_equal(q, engine=engine, check_row_order=check_row_order)


def test_unique_fallback(df, streaming_engine_factory):
    engine = streaming_engine_factory(
        StreamingOptions(
            unique_fraction={"y": 1.0},
            fallback_mode="raise",
            dynamic_planning=None,
        ),
    )
    q = df.unique(keep="first", maintain_order=True)
    with pytest.raises(
        NotImplementedError,
        match="Unsupported unique options",
    ):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("maintain_order", [True, False])
@pytest.mark.parametrize("cardinality", [{}, {"y": 0.5}])
def test_unique_select(df, streaming_engine_factory, maintain_order, cardinality):
    engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=4,
            unique_fraction=cardinality,
            fallback_mode="warn",
        ),
    )
    q = df.select(pl.col("y").unique(maintain_order=maintain_order))
    if cardinality == {"y": 0.5} and maintain_order:
        with pytest.warns(
            UserWarning, match="Unsupported unique options for multiple partitions."
        ):
            assert_gpu_result_equal(q, engine=engine, check_row_order=False)
    else:
        assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("keep", ["first", "last", "any"])
@pytest.mark.parametrize("zlice", ["head", "tail"])
def test_unique_head_tail(keep, zlice, streaming_engine_factory):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=4, fallback_mode="warn"),
    )
    data = [0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 9, 10]
    df = pl.LazyFrame({"x": data})
    q = df.unique(subset=None, keep=keep, maintain_order=True)
    expect = pl.LazyFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    # Cannot use assert_gpu_result_equal until CPU bug is fixed
    # See: https://github.com/pola-rs/polars/issues/22470
    assert_frame_equal(
        getattr(q, zlice)().collect(engine=engine),
        getattr(expect, zlice)().collect(engine=engine),
    )


def test_unique_complex_slice_fallback(df, engine):
    """Test that unique with complex slice (offset >= 1) falls back correctly."""
    # unique().slice(offset=5, length=10) has zlice[0] >= 1, triggering fallback
    q = df.unique(subset=("y",), keep="any").slice(5, 10)
    with pytest.warns(UserWarning, match="Complex slice not supported"):
        result = q.collect(engine=engine)
    # Just verify the fallback produces valid output with expected shape
    assert result.shape == (10, 3)

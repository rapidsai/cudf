# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine
from cudf_polars.testing.asserts import assert_gpu_result_equal

if TYPE_CHECKING:
    from collections.abc import Generator

    from rapidsmpf.communicator.communicator import Communicator

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine


@pytest.fixture
def engine(
    request: pytest.FixtureRequest,
    spmd_comm: Communicator,
) -> Generator[StreamingEngine, None, None]:
    params: dict[str, Any] = getattr(request, "param", {})
    executor_options = {
        "max_rows_per_partition": 50,
        "fallback_mode": "warn",
        "dynamic_planning": {},
        **params.get("executor_options", {}),
    }
    with SPMDEngine(comm=spmd_comm, executor_options=executor_options) as engine:
        yield engine


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
@pytest.mark.parametrize(
    "cardinality,engine",
    [
        ({}, {"executor_options": {"unique_fraction": {}}}),
        ({"y": 0.7}, {"executor_options": {"unique_fraction": {"y": 0.7}}}),
    ],
    indirect=["engine"],
)
def test_unique(df, keep, subset, maintain_order, cardinality, engine):
    q = df.unique(subset=subset, keep=keep, maintain_order=maintain_order)
    check_row_order = maintain_order
    if keep == "any" and subset:
        q = q.select(*(pl.col(col) for col in subset))
        check_row_order = False

    assert_gpu_result_equal(q, engine=engine, check_row_order=check_row_order)


@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {
                "unique_fraction": {"y": 1.0},
                "fallback_mode": "raise",
                "dynamic_planning": None,
            }
        }
    ],
    indirect=True,
)
def test_unique_fallback(df, engine):
    q = df.unique(keep="first", maintain_order=True)
    with pytest.raises(
        NotImplementedError,
        match="Unsupported unique options",
    ):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("maintain_order", [True, False])
@pytest.mark.parametrize(
    "cardinality,engine",
    [
        (
            {},
            {"executor_options": {"max_rows_per_partition": 4, "unique_fraction": {}}},
        ),
        (
            {"y": 0.5},
            {
                "executor_options": {
                    "max_rows_per_partition": 4,
                    "unique_fraction": {"y": 0.5},
                }
            },
        ),
    ],
    indirect=["engine"],
)
def test_unique_select(df, maintain_order, cardinality, engine):
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
@pytest.mark.parametrize(
    "engine",
    [{"executor_options": {"max_rows_per_partition": 4}}],
    indirect=True,
)
def test_unique_head_tail(keep, zlice, engine):
    data = [0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 9, 10]
    df = pl.LazyFrame({"x": data})
    q = df.unique(subset=None, keep=keep, maintain_order=True)
    expect = pl.LazyFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    # Cannot use assert_gpu_result_equal until CPU bug is fixed
    # See: https://github.com/pola-rs/polars/issues/22470
    assert_frame_equal(
        getattr(q, zlice)().collect(engine=engine),
        getattr(expect, zlice)().collect(),
    )


def test_unique_complex_slice_fallback(df, engine):
    """Test that unique with complex slice (offset >= 1) falls back correctly."""
    # unique().slice(offset=5, length=10) has zlice[0] >= 1, triggering fallback
    q = df.unique(subset=("y",), keep="any").slice(5, 10)
    with pytest.warns(UserWarning, match="Complex slice not supported"):
        result = q.collect(engine=engine)
    # Just verify the fallback produces valid output with expected shape
    assert result.shape == (10, 3)

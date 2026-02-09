# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RapidsMPF metadata functionality."""

from __future__ import annotations

import pytest
from rapidsmpf.streaming.cudf.channel_metadata import HashScheme

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.rapidsmpf.core import evaluate_logical_plan
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

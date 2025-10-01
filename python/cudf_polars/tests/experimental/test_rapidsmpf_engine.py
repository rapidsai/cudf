# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("fallback_mode", ["warn", "silent"])
@pytest.mark.parametrize("rows_per_partition", [1, 10, 20])
def test_rapidmpf_engine_fallback(rows_per_partition: int, fallback_mode: str) -> None:
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "scheduler": "rapidsmpf",
            "max_rows_per_partition": rows_per_partition,
            "fallback_mode": fallback_mode,
        },
    )
    df = pl.LazyFrame(
        {
            "a": list(range(20)),
            "b": list(range(20, 40)),
            "c": list(range(40, 60)),
            "d": list(range(60, 80)),
        }
    )
    q = df.select(pl.col("a") - (pl.col("b") + pl.col("c") * 2), pl.col("d")).sort("d")
    if rows_per_partition < 20 and fallback_mode == "warn":
        with pytest.raises(UserWarning):
            assert_gpu_result_equal(q, engine=engine)
    else:
        assert_gpu_result_equal(q, engine=engine)


def test_rapidmpf_engine_concat() -> None:
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "scheduler": "rapidsmpf",
            "max_rows_per_partition": 3,
        },
    )
    q = pl.concat(
        [
            pl.LazyFrame({"a": [1, 2, 3]}),
            pl.LazyFrame({"a": [4, 5, 6, 7, 8, 9]}),
        ]
    )
    assert_gpu_result_equal(q, engine=engine)

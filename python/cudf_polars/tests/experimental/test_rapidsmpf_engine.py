# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.rapidsmpf.core import lower_ir_graph
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.io import make_partitioned_source
from cudf_polars.utils.config import ConfigOptions

# Skip if rapidsmpf is not installed
pytest.importorskip("rapidsmpf")


@pytest.mark.parametrize("fallback_mode", ["warn", "silent"])
@pytest.mark.parametrize("rows_per_partition", [1, 10, 20])
def test_rapidmpf_engine_fallback(rows_per_partition: int, fallback_mode: str) -> None:
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "engine": "rapidsmpf",
            "max_rows_per_partition": rows_per_partition,
            "fallback_mode": fallback_mode,
            "scheduler": "synchronous",
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
            "engine": "rapidsmpf",
            "max_rows_per_partition": 3,
            "scheduler": "synchronous",
        },
    )
    q = pl.concat(
        [
            pl.LazyFrame({"a": [1, 2, 3]}),
            pl.LazyFrame({"a": [4, 5, 6, 7, 8, 9]}),
        ]
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("blocksize", [1_000, 10_000, 1_000_000])
@pytest.mark.parametrize("n_files", [2, 3])
def test_target_partition_size(tmp_path, blocksize, n_files):
    df = pl.DataFrame(
        {
            "x": range(3_000),
            "y": ["cat", "dog", "fish"] * 1_000,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 600,
        }
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=n_files)

    q = pl.scan_parquet(tmp_path)

    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "engine": "rapidsmpf",
            "target_partition_size": blocksize,
            "scheduler": "synchronous",
        },
    )
    assert_gpu_result_equal(q, engine=engine)

    # Check partitioning
    qir = Translator(q._ldf.visit(), engine).translate_ir()
    ir, info = lower_ir_graph(qir, ConfigOptions.from_polars_engine(engine))
    # NOTE: The first child is the Scan node.
    count = info[ir.children[0]].count
    if blocksize <= 12_000:
        assert count > n_files
    else:
        assert count < n_files

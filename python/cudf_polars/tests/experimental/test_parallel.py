# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars import Translator
from cudf_polars.experimental.parallel import evaluate_dask
from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_evaluate_dask():
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": [5, 6, 7], "d": [7, 9, 8]})

    q = df.select(pl.col("a") - (pl.col("b") + pl.col("c") * 2), pl.col("d")).sort("d")

    qir = Translator(q._ldf.visit()).translate_ir()

    expected = qir.evaluate(cache={}).to_polars()

    got = evaluate_dask(qir).to_polars()

    assert_frame_equal(expected, got)


def test_can_convert_lists():
    df = pl.LazyFrame(
        {
            "a": pl.Series([[1, 2], [3]], dtype=pl.List(pl.Int8())),
            "b": pl.Series([[1], [2]], dtype=pl.List(pl.UInt16())),
            "c": pl.Series(
                [
                    [["1", "2", "3"], ["4", "567"]],
                    [["8", "9"], []],
                ],
                dtype=pl.List(pl.List(pl.String())),
            ),
            "d": pl.Series([[[1, 2]], []], dtype=pl.List(pl.List(pl.UInt16()))),
        }
    )

    assert_gpu_result_equal(df, executor="dask-experimental")


def test_scan_csv_comment_char(tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n# 1,2,3\n3,4,5""")

    q = pl.scan_csv(tmp_path / "test.csv", comment_prefix="#")

    assert_gpu_result_equal(q, executor="dask-experimental")

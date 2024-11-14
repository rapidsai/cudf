# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars import Translator
from cudf_polars.experimental.parallel import evaluate_dask


def test_evaluate_dask():
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": [5, 6, 7], "d": [7, 9, 8]})

    q = df.select(pl.col("a") - (pl.col("b") + pl.col("c") * 2), pl.col("d")).sort("d")

    qir = Translator(q._ldf.visit()).translate_ir()

    expected = qir.evaluate(cache={}).to_polars()

    got = evaluate_dask(qir).to_polars()

    assert_frame_equal(expected, got)

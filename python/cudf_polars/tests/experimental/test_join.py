# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl


@pytest.mark.parametrize("how", ["inner", "left", "right"])
@pytest.mark.parametrize("num_rows_threshold", [5, 10, 15])
def test_parallel_join(how, num_rows_threshold):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
        parallel_options={"num_rows_threshold": num_rows_threshold},
    )
    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": ["cat", "dog", "fish"] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(6),
            "y": ["dog", "bird", "fish"] * 2,
            "zz": [1, 2] * 3,
        }
    )
    q = left.join(right, on="y", how=how)

    from cudf_polars import Translator
    from cudf_polars.experimental.parallel import evaluate_dask

    qir = Translator(q._ldf.visit(), engine).translate_ir()
    evaluate_dask(qir)

    # assert_gpu_result_equal(q, engine=engine, check_row_order=False)

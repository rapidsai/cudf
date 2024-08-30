# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import datetime

import polars as pl

from cudf_polars.testing.asserts import assert_ir_translation_raises


def test_groupby_dynamic_raises():
    df = pl.LazyFrame(
        {
            "dt": [
                datetime(2021, 12, 31, 0, 0, 0),
                datetime(2022, 1, 1, 0, 0, 1),
                datetime(2022, 3, 31, 0, 0, 1),
                datetime(2022, 4, 1, 0, 0, 1),
            ]
        }
    )

    q = (
        df.sort("dt")
        .group_by_dynamic("dt", every="1q")
        .agg(pl.col("dt").count().alias("num_values"))
    )
    assert_ir_translation_raises(q, NotImplementedError)

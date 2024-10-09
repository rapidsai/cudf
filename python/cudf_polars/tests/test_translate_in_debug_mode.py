# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl


def test_cudf_polars_debug_mode():
    df = pl.LazyFrame(
        {"key": [1, 1, 1, 2, 3, 3, 2, 2], "value": [1, 2, 3, 4, 5, 6, 7, 8]}
    )
    q = df.select(pl.col("value").sum().over("key"))

    with pytest.raises(
        Exception,
        match=r"(NotImplementedError|Query contained unsupported operations|Grouped rolling window not implemented)",
    ):
        q.collect(engine=pl.GPUEngine(debug_mode=True, raise_on_fail=True))
    # assert_ir_translation_raises(q, NotImplementedError)

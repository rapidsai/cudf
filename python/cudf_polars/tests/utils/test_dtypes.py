# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.utils.dtypes import from_polars


@pytest.mark.parametrize(
    "pltype",
    [
        pl.Time(),
        pl.Struct({"a": pl.Int8, "b": pl.Float32}),
        pl.Datetime("ms", time_zone="US/Pacific"),
        pl.List(pl.Datetime("ms", time_zone="US/Pacific")),
        pl.Array(pl.Int8, 2),
        pl.Binary(),
        pl.Categorical(),
        pl.Enum(["a", "b"]),
        pl.Field("a", pl.Int8),
        pl.Object(),
        pl.Unknown(),
    ],
    ids=repr,
)
def test_unhandled_dtype_conversion_raises(pltype):
    with pytest.raises(NotImplementedError):
        _ = from_polars(pltype)

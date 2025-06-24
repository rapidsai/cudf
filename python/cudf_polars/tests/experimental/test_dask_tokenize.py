# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dask.tokenize
import pytest

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl.expressions.base import Col, NamedExpr
from cudf_polars.experimental.dask_registers import register

# Must register sizeof dispatch before running tests
register()


@pytest.mark.parametrize(
    "value",
    [
        NamedExpr("a", Col(DataType(pl.Int64()), "a")),
        DataType(pl.Int64()),
    ],
    ids=["named_expr", "data_type"],
)
def test_tokenize(value: DataType | NamedExpr) -> None:
    normalizer = dask.tokenize.normalize_token.dispatch(type(value))
    package = normalizer.__module__.split(".")[0]
    assert package == "cudf_polars"

    dask.tokenize.tokenize(value)

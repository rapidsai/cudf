# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pyarrow as pa

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DType
from cudf_polars.dsl import expr
from cudf_polars.experimental.utils import _leaf_column_names


def test_leaf_column_names():
    dt = DType(pl.datatypes.Int32())
    a = expr.Col(dt, "a")
    b = expr.Literal(dt, pa.scalar(1, type=pa.int32()))
    c = expr.Col(dt, "c")
    d = expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, a, b)
    e = expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, d, c)
    assert set(_leaf_column_names(e)) == {"a", "c"}

# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pylibcudf as plc

from cudf_polars.dsl import expr
from cudf_polars.experimental.utils import _leaf_column_names


def test_leaf_column_names():
    dt = plc.DataType(plc.TypeId.INT32)
    a = expr.Col(dt, "a")
    b = expr.Col(dt, "b")
    c = expr.Col(dt, "c")
    d = expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, a, b)
    e = expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, d, c)
    assert set(_leaf_column_names(e)) == {"a", "b", "c"}

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import polars as pl

from cudf_polars.dsl.expr import Expr
import cudf._lib.pylibcudf as plc


class NotImplementedExpr(Expr):
    def do_evaluate(self, df, *, context, mapping):
        raise NotImplementedError("Example Unsupported cuDF operation")


def invoke_unimplemented_op(df):
    # TODO: does not work
    # df.select(NotImplementedExpr(plc.DataType(plc.TypeId.INT64)))
    pass


def test_polars_verbose_warns():
    old_val = os.environ.get("POLARS_VERBOSE")
    os.environ["POLARS_VERBOSE"] = "1"
    df = pl.DataFrame({"a": [1, 2, 3]}).lazy()

    invoke_unimplemented_op(df)

    os.environ["POLARS_VERBOSE"] = old_val

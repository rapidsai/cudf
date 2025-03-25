# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.containers import DataFrame
from cudf_polars.dsl import expr


def test_evaluate_mapping():
    pdf = pl.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    df = DataFrame.from_polars(pdf)

    cola = expr.Col(df.column_map["a"].obj.type(), "a")
    colb = expr.Col(df.column_map["b"].obj.type(), "b")

    result = DataFrame([cola.evaluate(df).rename("a")]).to_polars()
    assert_frame_equal(result, pdf.select("a"))

    mapping = {cola: df.column_map["b"]}
    result = DataFrame(
        [
            colb.evaluate(df, mapping=mapping).rename("colb"),
            cola.evaluate(df, mapping=mapping).rename("cola_mapped"),
        ]
    ).to_polars()

    assert_frame_equal(result, pdf.select(colb="b", cola_mapped="b"))

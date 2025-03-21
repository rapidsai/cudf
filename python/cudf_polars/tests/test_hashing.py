# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

import cudf_polars.dsl._hashing as hashing
from cudf_polars.dsl.translate import Translator


def test_hash_polars_dataframe():
    # Exercise all the data types we care about
    df = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, 3], dtype=pl.Int64),
            # "b": pl.Series(['a', 'b', 'c'], dtype=pl.Utf8),
            "c": pl.Series([True, False, True], dtype=pl.Boolean),
            "d": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
            "e": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
            # "g": pl.Series([1.0, 2.0, 3.0], dtype=pl.Decimal),
            "h": pl.Series([1, 2, 3], dtype=pl.Duration),
            "i": pl.Series([1, 2, 3], dtype=pl.Datetime),
            "j": pl.Series([1, 2, 3], dtype=pl.Date),
            # "k": pl.Series([1, 2, 3], dtype=pl.Time),
            "l": pl.Series([1, 2, 3], dtype=pl.Int8),
            "m": pl.Series([1, 2, 3], dtype=pl.Int16),
            "n": pl.Series([1, 2, 3], dtype=pl.Int32),
            "o": pl.Series([1, 2, 3], dtype=pl.Int64),
            "p": pl.Series([1, 2, 3], dtype=pl.UInt8),
            "q": pl.Series([1, 2, 3], dtype=pl.UInt16),
            "r": pl.Series([1, 2, 3], dtype=pl.UInt32),
            "s": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "t": pl.Series([[1], [2], [3]], dtype=pl.List(pl.Int64)),
            # "u": pl.Series([{"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}, {"a": 3, "b": "baz"}], dtype=pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Utf8)])),
            # "v": pl.Series([b"foo", b"bar", b"baz"], dtype=pl.Binary),
        }
    )

    # ths is the easiest way to get the internal PyDataFrame object.
    ir = Translator(df._ldf.visit(), pl.GPUEngine()).translate_ir()
    pydf = ir.df

    h = hashing.hash_polars_dataframe(pydf)
    assert h == hashing.hash_polars_dataframe(pydf)

    # subsetting columns produces a different hash
    assert hashing.hash_polars_dataframe(pydf.select(["a", "c"])) != h

    # # adding a new column produces a different hash
    # assert hashing.hash_polars_dataframe(pydf.with_columns([pl.lit(1).alias("z")])) != h

    # changing the order of columns produces a different hash
    assert (
        hashing.hash_polars_dataframe(pydf.select(list(reversed(pydf.columns())))) != h
    )

    # subsetting the rows produces a different hash
    # this is failing... Doing just the row count isn't enough
    # since we could have equal-length slices into different parts
    # of the dataframe.
    assert hashing.hash_polars_dataframe(pydf.slice(1, 2)) != h

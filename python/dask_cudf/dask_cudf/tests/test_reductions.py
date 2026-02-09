# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import dask
from dask import dataframe as dd

import cudf

import dask_cudf
from dask_cudf.tests.utils import _make_random_frame

_reducers = ["sum", "count", "mean", "var", "std", "min", "max"]

_NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "Dan",
    "Edith",
    "Frank",
    "George",
    "Hannah",
    "Ingrid",
    "Jerry",
    "Kevin",
    "Laura",
    "Michael",
    "Norbert",
    "Oliver",
    "Patricia",
    "Quinn",
    "Ray",
    "Sarah",
    "Tim",
    "Ursula",
    "Victor",
    "Wendy",
    "Xavier",
    "Yvonne",
    "Zelda",
]


def _get_reduce_fn(name):
    def wrapped(series):
        fn = getattr(series, name)
        return fn()

    return wrapped


@pytest.mark.parametrize("reducer", _reducers)
def test_series_reduce(reducer):
    reducer = _get_reduce_fn(reducer)
    size = 10
    df, gdf = _make_random_frame(size)

    got = reducer(gdf.x)
    exp = reducer(df.x)
    dd.assert_eq(got, exp)


@pytest.mark.parametrize(
    "data",
    [
        (
            lambda r=np.random.default_rng(0): cudf.from_pandas(
                pd.DataFrame(
                    {
                        "a": pd.Categorical.from_codes(
                            r.integers(0, len(_NAMES), 10000), _NAMES
                        ),
                        "b": r.poisson(1000, 10000),
                        "c": r.random(10000) * 2 - 1,
                        "d": r.poisson(1000, 10000),
                    },
                    columns=["a", "b", "c", "d"],
                )
            )
        )(),
        (
            lambda r=np.random.default_rng(0): cudf.from_pandas(
                pd.DataFrame(
                    {
                        "a": pd.Categorical.from_codes(
                            r.integers(0, len(_NAMES), 10000), _NAMES
                        ),
                        "b": r.poisson(1000, 10000),
                        "c": r.random(10000) * 2 - 1,
                        "d": r.choice(_NAMES, 10000),
                    },
                    columns=["a", "b", "c", "d"],
                )
            )
        )(),
        (
            lambda r=np.random.default_rng(0): cudf.from_pandas(
                pd.DataFrame(
                    {
                        "a": r.choice([True, False], 10000),
                        "b": r.poisson(1000, 10000),
                        "c": r.random(10000) * 2 - 1,
                        "d": r.choice(_NAMES, 10000),
                    },
                    columns=["a", "b", "c", "d"],
                )
            )
        )(),
    ],
)
@pytest.mark.parametrize(
    "op", ["max", "min", "sum", "prod", "mean", "var", "std"]
)
def test_rowwise_reductions(data, op):
    gddf = dask_cudf.from_cudf(data, npartitions=10)
    pddf = gddf.to_backend("pandas")

    with dask.config.set({"dataframe.convert-string": False}):
        if op in ("var", "std"):
            expected = getattr(pddf, op)(axis=1, numeric_only=True, ddof=0)
            got = getattr(gddf, op)(axis=1, numeric_only=True, ddof=0)
        else:
            expected = getattr(pddf, op)(numeric_only=True, axis=1)
            got = getattr(pddf, op)(numeric_only=True, axis=1)

        dd.assert_eq(
            expected,
            got,
            check_exact=False,
            check_dtype=op not in ("var", "std"),
        )


@pytest.mark.parametrize("skipna", [True, False])
def test_var_nulls(skipna):
    # Copied from 10min example notebook
    # See: https://github.com/rapidsai/cudf/pull/15347
    s = cudf.Series([1, 2, 3, None, 4])
    ds = dask_cudf.from_cudf(s, npartitions=2)
    dd.assert_eq(s.var(skipna=skipna), ds.var(skipna=skipna))
    dd.assert_eq(s.std(skipna=skipna), ds.std(skipna=skipna))

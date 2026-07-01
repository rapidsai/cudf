# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [],
        [0, 12, 14],
        [0, 14, 12, 12, 3, 10, 12, 14],
        np.random.default_rng(seed=0).integers(-100, 100, 200),
        pd.Series([0.0, 1.0, None, 10.0]),
        [None, None, None, None],
        [np.nan, None, -1, 2, 3],
        [1, 2],
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        np.random.default_rng(seed=0).integers(-100, 100, 10),
        [],
        [np.nan, None, -1, 2, 3],
        [1.0, 12.0, None, None, 120],
        [0.1, 12.1, 14.1],
        [0, 14, 12, 12, 3, 10, 12, 14, None],
        [None, None, None],
        ["0", "12", "14"],
        ["0", "12", "14", "a"],
        [1.0, 2.5],
    ],
)
def test_isin_numeric(data, values):
    rng = np.random.default_rng(seed=0)
    index = rng.integers(0, 100, len(data))
    psr = pd.Series(data, index=index)
    gsr = cudf.Series(psr, nan_as_null=False)

    expected = psr.isin(values)
    got = gsr.isin(values)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(
            ["2018-01-01", "2019-04-03", None, "2019-12-30"],
            dtype="datetime64[ns]",
        ),
        pd.Series(
            [
                "2018-01-01",
                "2019-04-03",
                None,
                "2019-12-30",
                "2018-01-01",
                "2018-01-01",
            ],
            dtype="datetime64[ns]",
        ),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        [1514764800000000000, 1577664000000000000],
        [
            1514764800000000000,
            1577664000000000000,
            1577664000000000000,
            1577664000000000000,
            1514764800000000000,
        ],
        ["2019-04-03", "2019-12-30", "2012-01-01"],
        [
            "2012-01-01",
            "2012-01-01",
            "2012-01-01",
            "2019-04-03",
            "2019-12-30",
            "2012-01-01",
        ],
    ],
)
def test_isin_datetime(data, values):
    psr = pd.Series(data)
    gsr = cudf.Series(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        ["this", "is", None, "a", "test"],
        ["test", "this", "test", "is", None, "test", "a", "test"],
        ["0", "12", "14"],
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["this", "is"],
        [None, None, None],
        ["12", "14", "19"],
        [12, 14, 19],
        ["is", "this", "is", "this", "is"],
    ],
)
def test_isin_string(data, values):
    psr = pd.Series(data)
    gsr = cudf.Series(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(["a", "b", "c", "c", "c", "d", "e"], dtype="category"),
        pd.Series(["a", "b", None, "c", "d", "e"], dtype="category"),
        pd.Series([0, 3, 10, 12], dtype="category"),
        pd.Series([0, 3, 10, 12, 0, 10, 3, 0, 0, 3, 3], dtype="category"),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["a", "b", None, "f", "words"],
        ["0", "12", None, "14"],
        [0, 10, 12, None, 39, 40, 1000],
        [0, 0, 0, 0, 3, 3, 3, None, 1, 2, 3],
    ],
)
def test_isin_categorical(data, values):
    psr = pd.Series(data)
    gsr = cudf.Series(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
@pytest.mark.parametrize(
    "data,values",
    [
        ([0, 1, 0], [1]),
        ([0, 1, 0], [1, pd.NA]),
        ([0, pd.NA, 0], [1, 0]),
        ([0, 1, pd.NA], [1, pd.NA]),
        ([0, 1, pd.NA], [1, np.nan]),
        ([0, pd.NA, pd.NA], [np.nan, pd.NaT, None]),
    ],
)
def test_isin_masked_types(dtype, data, values):
    # Series.isin on a pandas masked (nullable integer/float/boolean) dtype
    # returns a nullable BooleanDtype result and matches pandas' NA semantics:
    #  * comparison is done on the underlying values (a boolean element equals
    #    the integer 1), and
    #  * an NA element is considered present only when pd.NA itself is one of
    #    ``values`` (a plain NaN/None/NaT does not match).
    psr = pd.Series(data, dtype=dtype)
    gsr = cudf.Series(data, dtype=dtype)

    got = gsr.isin(values)
    expected = psr.isin(values)

    assert got.dtype == pd.BooleanDtype()
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "values",
    [[1], [0], [1, 0], [2], [1.0], [1.5], [True], [False]],
)
def test_isin_bool_against_numeric(values):
    # A boolean Series compares equal to the integers/floats 0 and 1, matching
    # numpy/pandas value semantics (previously cudf returned all-False for a
    # numeric ``values`` argument).
    psr = pd.Series([True, False, False, True])
    gsr = cudf.Series([True, False, False, True])

    got = gsr.isin(values)
    assert got.dtype == np.dtype("bool")
    assert_eq(got, psr.isin(values))


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series([0, 1, 0], dtype=pd.ArrowDtype(pa.int64())),
        pd.Series([1.0, 2.0, None], dtype=pd.ArrowDtype(pa.float64())),
        pd.Series([True, False, True], dtype=pd.ArrowDtype(pa.bool_())),
        pd.Series(["a", "b", "a"], dtype="category"),
    ],
)
def test_isin_non_masked_extension_returns_numpy_bool(psr):
    # Arrow and categorical inputs yield a numpy bool result (only masked
    # numeric/boolean dtypes upgrade to nullable boolean).
    gsr = cudf.from_pandas(psr)

    got = gsr.isin([psr.iloc[0]])
    assert got.dtype == np.dtype("bool")
    assert_eq(got, psr.isin([psr.iloc[0]]))

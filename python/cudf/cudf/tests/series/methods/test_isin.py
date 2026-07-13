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


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype("object"),
        pd.StringDtype(storage="python", na_value=pd.NA),
        pd.StringDtype(storage="python", na_value=np.nan),
        pd.StringDtype(storage="pyarrow", na_value=pd.NA),
        pd.StringDtype(storage="pyarrow", na_value=np.nan),
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_string()),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        ["b", "x"],
        np.array(["b", "x"], dtype=object),
        pd.array(
            ["b", "x"], dtype=pd.StringDtype(storage="python", na_value=pd.NA)
        ),
        pd.array(
            ["b", "x"],
            dtype=pd.StringDtype(storage="python", na_value=np.nan),
        ),
        pd.array(
            ["b", "x"],
            dtype=pd.StringDtype(storage="pyarrow", na_value=pd.NA),
        ),
        pd.array(
            ["b", "x"],
            dtype=pd.StringDtype(storage="pyarrow", na_value=np.nan),
        ),
        pd.array(["b", "x"], dtype=pd.ArrowDtype(pa.string())),
        pa.array(["b", "x"]),
    ],
)
def test_isin_string_dtype_flavors(dtype, values):
    # isin must match on element values regardless of which string dtype
    # flavor the series and the values use (object, pd.StringDtype with
    # python/pyarrow storage and NaN/NA na_value, pd.ArrowDtype string).
    data = ["a", "b", "c"]
    psr = pd.Series(data, dtype=dtype)
    gsr = cudf.Series(data, dtype=dtype)

    got = gsr.isin(values)
    if isinstance(values, pa.Array) and (
        dtype == np.dtype("object")
        or (isinstance(dtype, pd.StringDtype) and dtype.storage == "python")
    ):
        # pandas does not match pyarrow.Array values against
        # object/python-storage series (returns all-False); cudf matches
        # on element values.
        assert got.to_pandas().tolist() == [False, True, False]
    else:
        expected = psr.isin(values)
        assert_eq(got, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype("object"),
        pd.StringDtype(storage="python", na_value=pd.NA),
        pd.StringDtype(storage="python", na_value=np.nan),
        pd.StringDtype(storage="pyarrow", na_value=pd.NA),
        pd.StringDtype(storage="pyarrow", na_value=np.nan),
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_string()),
    ],
)
def test_isin_string_null_values(dtype):
    # cudf treats every NA-like value in ``values`` (None, pd.NA) as a
    # match for nulls in the series. Results are compared with pandas
    # except where pandas diverges: object dtype distinguishes the
    # stored None from pd.NA, and ArrowDtype raises ArrowTypeError when
    # the value set contains nulls.
    data = ["a", "b", None]
    psr = pd.Series(data, dtype=dtype)
    gsr = cudf.Series(data, dtype=dtype)

    got = gsr.isin([None])
    if isinstance(dtype, pd.ArrowDtype):
        assert got.to_pandas().tolist() == [False, False, True]
    else:
        assert_eq(got, psr.isin([None]))

    got = gsr.isin(["a", pd.NA])
    if isinstance(dtype, pd.ArrowDtype) or dtype == np.dtype("object"):
        assert got.to_pandas().tolist() == [True, False, True]
    else:
        assert_eq(got, psr.isin(["a", pd.NA]))

    assert_eq(gsr.isin([]), psr.isin([]))

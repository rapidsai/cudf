# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def test_select_dtype():
    gdf = cudf.datasets.randomdata(
        nrows=20, dtypes={"a": "category", "b": int, "c": float, "d": str}
    )
    pdf = gdf.to_pandas()

    assert_eq(pdf.select_dtypes("float64"), gdf.select_dtypes("float64"))
    assert_eq(pdf.select_dtypes(np.float64), gdf.select_dtypes(np.float64))
    assert_eq(
        pdf.select_dtypes(include=["float64"]),
        gdf.select_dtypes(include=["float64"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["object", "int", "category"]),
        gdf.select_dtypes(include=["object", "int", "category"]),
    )

    assert_eq(
        pdf.select_dtypes(include=["int64", "float64"]),
        gdf.select_dtypes(include=["int64", "float64"]),
    )
    assert_eq(
        pdf.select_dtypes(include=np.number),
        gdf.select_dtypes(include=np.number),
    )
    assert_eq(
        pdf.select_dtypes(include=[np.int64, np.float64]),
        gdf.select_dtypes(include=[np.int64, np.float64]),
    )

    assert_eq(
        pdf.select_dtypes(include=["category"]),
        gdf.select_dtypes(include=["category"]),
    )
    assert_eq(
        pdf.select_dtypes(exclude=np.number),
        gdf.select_dtypes(exclude=np.number),
    )

    assert_exceptions_equal(
        lfunc=pdf.select_dtypes,
        rfunc=gdf.select_dtypes,
        lfunc_args_and_kwargs=([], {"includes": ["Foo"]}),
        rfunc_args_and_kwargs=([], {"includes": ["Foo"]}),
    )

    assert_exceptions_equal(
        lfunc=pdf.select_dtypes,
        rfunc=gdf.select_dtypes,
        lfunc_args_and_kwargs=(
            [],
            {"exclude": np.number, "include": np.number},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"exclude": np.number, "include": np.number},
        ),
    )

    gdf = cudf.DataFrame(
        {"A": [3, 4, 5], "C": [1, 2, 3], "D": ["a", "b", "c"]}
    )
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes(include=["object", "int", "category"]),
        gdf.select_dtypes(include=["object", "int", "category"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["object"], exclude=["category"]),
        gdf.select_dtypes(include=["object"], exclude=["category"]),
    )

    gdf = cudf.DataFrame({"a": range(10), "b": range(10, 20)})
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes(include=["category"]),
        gdf.select_dtypes(include=["category"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["float"]),
        gdf.select_dtypes(include=["float"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["object"]),
        gdf.select_dtypes(include=["object"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["int"]), gdf.select_dtypes(include=["int"])
    )
    assert_eq(
        pdf.select_dtypes(exclude=["float"]),
        gdf.select_dtypes(exclude=["float"]),
    )
    assert_eq(
        pdf.select_dtypes(exclude=["object"]),
        gdf.select_dtypes(exclude=["object"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["int"], exclude=["object"]),
        gdf.select_dtypes(include=["int"], exclude=["object"]),
    )

    assert_exceptions_equal(
        lfunc=pdf.select_dtypes,
        rfunc=gdf.select_dtypes,
    )

    gdf = cudf.DataFrame(
        {"a": cudf.Series([], dtype="int"), "b": cudf.Series([], dtype="str")}
    )
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes(exclude=["object"]),
        gdf.select_dtypes(exclude=["object"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["int"], exclude=["object"]),
        gdf.select_dtypes(include=["int"], exclude=["object"]),
    )

    gdf = cudf.DataFrame(
        {"int_col": [0, 1, 2], "list_col": [[1, 2], [3, 4], [5, 6]]}
    )
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes("int64"),
        gdf.select_dtypes("int64"),
    )


def test_select_dtype_datetime():
    gdf = cudf.datasets.timeseries(
        start="2000-01-01", end="2000-01-02", freq="3600s", dtypes={"x": int}
    )
    gdf = gdf.reset_index()
    pdf = gdf.to_pandas()

    assert_eq(pdf.select_dtypes("datetime64"), gdf.select_dtypes("datetime64"))
    assert_eq(
        pdf.select_dtypes(np.dtype("datetime64")),
        gdf.select_dtypes(np.dtype("datetime64")),
    )
    assert_eq(
        pdf.select_dtypes(include="datetime64"),
        gdf.select_dtypes(include="datetime64"),
    )


def test_select_dtype_datetime_with_frequency():
    gdf = cudf.datasets.timeseries(
        start="2000-01-01", end="2000-01-02", freq="3600s", dtypes={"x": int}
    )
    gdf = gdf.reset_index()
    pdf = gdf.to_pandas()

    assert_exceptions_equal(
        pdf.select_dtypes,
        gdf.select_dtypes,
        (["datetime64[ms]"],),
        (["datetime64[ms]"],),
    )

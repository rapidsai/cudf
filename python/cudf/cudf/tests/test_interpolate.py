# Copyright (c) 2021-2022, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing._utils import assert_eq, assert_exceptions_equal


@pytest.mark.parametrize(
    "data",
    [
        # basics
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]},
        {"A": [1.0, None, 3.0], "B": [4.0, None, 6.0]},
        {"A": [None, 2.0, 3.0], "B": [4.0, 5.0, None]},
    ],
)
@pytest.mark.parametrize("method", ["linear"])
@pytest.mark.parametrize("axis", [0])
def test_interpolate_dataframe(data, method, axis):
    # Pandas interpolate methods do not seem to work
    # with nullable dtypes yet, so this method treats
    # NAs as NaNs
    # https://github.com/pandas-dev/pandas/issues/40252
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    expect = pdf.interpolate(method=method, axis=axis)
    got = gdf.interpolate(method=method, axis=axis)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0],
        [1.0, None, 3.0],
        [None, 2.0, None, 4.0],
        [1.0, None, 3.0, None],
        [None, None, 3.0, 4.0],
        [1.0, 2.0, None, None],
        [None, None, None, None],
        [0.1, 0.2, 0.3],
    ],
)
@pytest.mark.parametrize("method", ["linear"])
@pytest.mark.parametrize("axis", [0])
def test_interpolate_series(data, method, axis):
    gsr = cudf.Series(data)
    psr = gsr.to_pandas()

    expect = psr.interpolate(method=method, axis=axis)
    got = gsr.interpolate(method=method, axis=axis)

    assert_eq(expect, got, check_dtype=psr.dtype != "object")


@pytest.mark.parametrize(
    "data,index", [([2.0, None, 4.0, None, 2.0], [1, 2, 3, 2, 1])]
)
def test_interpolate_series_unsorted_index(data, index):
    gsr = cudf.Series(data, index=index)
    psr = gsr.to_pandas()

    expect = psr.interpolate(method="values")
    got = gsr.interpolate(method="values")

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0, 4.0],
        [None, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, None],
        [None, None, 3.0, 4.0],
        [1.0, 2.0, None, None],
        [1.0, None, 3.0, None],
        [None, 2.0, None, 4.0],
        [None, None, None, None],
    ],
)
@pytest.mark.parametrize("index", [[0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 4, 9]])
@pytest.mark.parametrize("method", ["index", "values"])
def test_interpolate_series_values_or_index(data, index, method):
    gsr = cudf.Series(data, index=index)
    psr = gsr.to_pandas()

    expect = psr.interpolate(method=method)
    got = gsr.interpolate(method=method)

    assert_eq(expect, got, check_dtype=psr.dtype != "object")


@pytest.mark.parametrize(
    "data,kwargs",
    [
        (
            {"A": ["a", "b", "c"], "B": ["d", "e", "f"]},
            {"axis": 0, "method": "linear"},
        ),
        ({"A": [1, 2, 3]}, {"method": "pad", "limit_direction": "backward"}),
        ({"A": [1, 2, 3]}, {"method": "ffill", "limit_direction": "backward"}),
        ({"A": [1, 2, 3]}, {"method": "bfill", "limit_direction": "forward"}),
        (
            {"A": [1, 2, 3]},
            {"method": "backfill", "limit_direction": "forward"},
        ),
    ],
)
def test_interpolate_dataframe_error_cases(data, kwargs):
    gsr = cudf.DataFrame(data)
    psr = gsr.to_pandas()

    assert_exceptions_equal(
        lfunc=psr.interpolate,
        rfunc=gsr.interpolate,
        lfunc_args_and_kwargs=([], kwargs),
        rfunc_args_and_kwargs=([], kwargs),
    )

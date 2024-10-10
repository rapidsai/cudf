# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1.0, 2, None, 4],
        ["one", "two", "three", "four"],
        pd.Series(["a", "b", "c", "d"], dtype="category"),
        pd.Series(pd.date_range("2010-01-01", "2010-01-04")),
    ],
)
@pytest.mark.parametrize("nulls", ["one", "some", "all", "none"])
@pytest.mark.parametrize("inplace", [True, False])
def test_dropna_series(data, nulls, inplace):
    psr = pd.Series(data)
    rng = np.random.default_rng(seed=0)
    if len(data) > 0:
        if nulls == "one":
            p = rng.integers(0, 4)
            psr[p] = None
        elif nulls == "some":
            p1, p2 = rng.integers(0, 4, (2,))
            psr[p1] = None
            psr[p2] = None
        elif nulls == "all":
            psr[:] = None

    gsr = cudf.from_pandas(psr)

    check_dtype = True
    if gsr.null_count == len(gsr):
        check_dtype = False

    expected = psr.dropna()
    actual = gsr.dropna()

    if inplace:
        expected = psr
        actual = gsr

    assert_eq(expected, actual, check_dtype=check_dtype)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, None]},
        {"a": [1, 2, None], "b": [3, 4, 5]},
        {"a": [1, 2, None], "b": [3, 4, None]},
        {"a": [None, 1, 2], "b": [1, 2, None]},
        {"a": [None, 1, None], "b": [None, 2, None]},
        {"a": [None, None, 1], "b": [1, 2, None]},
        {"a": ["d", "e", "f"], "b": ["a", None, "c"]},
    ],
)
@pytest.mark.parametrize("how", ["all", "any"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("inplace", [True, False])
def test_dropna_dataframe(data, how, axis, inplace):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    expected = pdf.dropna(axis=axis, how=how, inplace=inplace)
    actual = gdf.dropna(axis=axis, how=how, inplace=inplace)

    if inplace:
        expected = pdf
        actual = gdf

    assert_eq(expected, actual)


@pytest.mark.parametrize("how", ["all", "any"])
@pytest.mark.parametrize(
    "data",
    [
        {
            "a": cudf.Series([None, None, None], dtype="float64"),
            "b": cudf.Series([1, 2, None]),
        },
        {
            "a": cudf.Series([np.nan, np.nan, np.nan], dtype="float64"),
            "b": cudf.Series([1, 2, None]),
        },
        cudf.Series([None, None, None], dtype="object"),
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
def test_dropna_with_all_nulls(how, data, axis):
    gdf = cudf.DataFrame({"a": data})
    pdf = gdf.to_pandas()

    assert_eq(pdf.dropna(axis=axis, how=how), gdf.dropna(axis=axis, how=how))


def test_dropna_nan_as_null():
    sr = cudf.Series([1.0, 2.0, np.nan, None], nan_as_null=False)
    assert_eq(sr.dropna(), sr[:2])
    sr = sr.nans_to_nulls()
    assert_eq(sr.dropna(), sr[:2])

    df = cudf.DataFrame(
        {
            "a": cudf.Series([1.0, 2.0, np.nan, None], nan_as_null=False),
            "b": cudf.Series([1, 2, 3, 4]),
        }
    )

    got = df.dropna()
    expected = df[:2]
    assert_eq(expected, got)

    df = df.nans_to_nulls()
    got = df.dropna()
    expected = df[:2]
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data,subset",
    [
        ({"a": [1, None], "b": [1, 2]}, ["a"]),
        ({"a": [1, None], "b": [1, 2]}, ["b"]),
        ({"a": [1, None], "b": [1, 2]}, []),
        ({"a": [1, 2], "b": [1, 2]}, ["b"]),
        ({"a": [1, 2, None], "b": [1, None, 2]}, ["a"]),
        ({"a": [1, 2, None], "b": [1, None, 2]}, ["b"]),
        ({"a": [1, 2, None], "b": [1, None, 2]}, ["a", "b"]),
    ],
)
def test_dropna_subset_rows(data, subset):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.dropna(subset=subset), gdf.dropna(subset=subset))


@pytest.mark.parametrize(
    "data, subset",
    [
        ({"a": [1, None], "b": [1, 2]}, [0]),
        ({"a": [1, None], "b": [1, 2]}, [1]),
        ({"a": [1, None], "b": [1, 2]}, []),
        ({"a": [1, 2], "b": [1, 2]}, [0]),
        ({"a": [1, 2], "b": [None, 2], "c": [3, None]}, [0]),
        ({"a": [1, 2], "b": [None, 2], "c": [3, None]}, [1]),
        ({"a": [1, 2], "b": [None, 2], "c": [3, None]}, [0, 1]),
    ],
)
def test_dropna_subset_cols(data, subset):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.dropna(axis=1, subset=subset), gdf.dropna(axis=1, subset=subset)
    )


# TODO: can't test with subset=[] below since Pandas
# returns empty DF when both subset=[] and thresh are specified.
@pytest.mark.parametrize("thresh", [0, 1, 2])
@pytest.mark.parametrize("subset", [None, ["a"], ["b"], ["a", "b"]])
def test_dropna_thresh(thresh, subset):
    pdf = pd.DataFrame({"a": [1, 2, None, None], "b": [1, 2, 3, None]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.dropna(axis=0, thresh=thresh, subset=subset),
        gdf.dropna(axis=0, thresh=thresh, subset=subset),
    )


@pytest.mark.parametrize("thresh", [0, 1, 2])
@pytest.mark.parametrize("subset", [None, [0], [1], [0, 1]])
@pytest.mark.parametrize("inplace", [True, False])
def test_dropna_thresh_cols(thresh, subset, inplace):
    pdf = pd.DataFrame(
        {"a": [1, 2], "b": [3, 4], "c": [5, None], "d": [np.nan, np.nan]}
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.dropna(
        axis=1, thresh=thresh, subset=subset, inplace=inplace
    )
    actual = gdf.dropna(axis=1, thresh=thresh, subset=subset, inplace=inplace)

    if inplace:
        expected = pdf
        actual = gdf

    assert_eq(
        expected,
        actual,
    )


@pytest.mark.parametrize(
    "data",
    [
        {
            "key": [1, 2, 10],
            "val": cudf.Series([np.nan, 3, 1], nan_as_null=False),
            "abc": [np.nan, None, 1],
        },
        {
            "key": [None, 2, 1],
            "val": cudf.Series([3, np.nan, 0.1], nan_as_null=True),
            "abc": [None, 1, None],
        },
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
def test_dropna_dataframe_np_nan(data, axis):
    gdf = cudf.DataFrame(data)
    pd_data = {
        key: value.to_pandas() if isinstance(value, cudf.Series) else value
        for key, value in data.items()
    }
    pdf = pd.DataFrame(pd_data)

    assert_eq(pdf.dropna(axis=axis), gdf.dropna(axis=axis), check_dtype=False)


@pytest.mark.parametrize(
    "data, dtype",
    [
        ([1, float("nan"), 2], "float64"),
        (["x", None, "y"], "str"),
        (["x", None, "y"], "category"),
        (["2020-01-20", pd.NaT, "2020-03-15"], "datetime64[ns]"),
        (["1s", pd.NaT, "3d"], "timedelta64[ns]"),
    ],
)
def test_dropna_index(data, dtype):
    pi = pd.Index(data, dtype=dtype)
    gi = cudf.from_pandas(pi)

    expect = pi.dropna()
    got = gi.dropna()

    assert_eq(expect, got)


@pytest.mark.parametrize("data", [[[1, None, 2], [None, None, 2]]])
@pytest.mark.parametrize("how", ["all", "any"])
def test_dropna_multiindex(data, how):
    pi = pd.MultiIndex.from_arrays(data)
    gi = cudf.from_pandas(pi)

    expect = pi.dropna(how)
    got = gi.dropna(how)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [
            [pd.Timestamp("2020-01-01"), pd.NaT, pd.Timestamp("2020-02-01")],
            [pd.NaT, pd.NaT, pd.Timestamp("2020-03-01")],
        ],
        [
            [pd.Timestamp("2020-01-01"), pd.NaT, pd.Timestamp("2020-02-01")],
            [np.nan, np.nan, 1.0],
        ],
        [[1.0, np.nan, 2.0], [np.nan, np.nan, 1.0]],
    ],
)
@pytest.mark.parametrize("how", ["all", "any"])
def test_dropna_multiindex_2(data, how):
    pi = pd.MultiIndex.from_arrays(data)
    gi = cudf.from_pandas(pi)

    expect = pi.dropna(how)
    got = gi.dropna(how)

    assert_eq(expect, got)


def test_ignore_index():
    pser = pd.Series([1, 2, np.nan], index=[2, 4, 1])
    gser = cudf.from_pandas(pser)

    result = pser.dropna(ignore_index=True)
    expected = gser.dropna(ignore_index=True)
    assert_eq(result, expected)

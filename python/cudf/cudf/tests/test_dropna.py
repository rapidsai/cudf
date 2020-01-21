import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


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
def test_dropna_series(data, nulls):

    psr = pd.Series(data)

    if len(data) > 0:
        if nulls == "one":
            p = np.random.randint(0, 4)
            psr[p] = None
        elif nulls == "some":
            p1, p2 = np.random.randint(0, 4, (2,))
            psr[p1] = None
            psr[p2] = None
        elif nulls == "all":
            psr[:] = None

    gsr = cudf.from_pandas(psr)

    check_dtype = True
    if gsr.null_count == len(gsr):
        check_dtype = False

    assert_eq(psr.dropna(), gsr.dropna(), check_dtype=check_dtype)


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
def test_dropna_dataframe(data, how, axis):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.dropna(axis=axis, how=how), gdf.dropna(axis=axis, how=how))


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
    assert_eq(sr.dropna(), sr[:3])
    sr = sr.nans_to_nulls()
    assert_eq(sr.dropna(), sr[:2])

    df = cudf.DataFrame(
        {
            "a": cudf.Series([1.0, 2.0, np.nan, None], nan_as_null=False),
            "b": cudf.Series([1, 2, 3, 4]),
        }
    )

    got = df.dropna()
    expected = df[:3]
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
def test_dropna_thresh_cols(thresh, subset):
    pdf = pd.DataFrame(
        {"a": [1, 2], "b": [3, 4], "c": [5, None], "d": [np.nan, np.nan]}
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.dropna(axis=1, thresh=thresh, subset=subset),
        gdf.dropna(axis=1, thresh=thresh, subset=subset),
    )

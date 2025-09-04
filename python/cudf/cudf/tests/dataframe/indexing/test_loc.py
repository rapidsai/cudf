# Copyright (c) 2025, NVIDIA CORPORATION.
import re
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


def test_dataframe_midx_columns_loc():
    idx_1 = ["Hi", "Lo"]
    idx_2 = ["I", "II", "III"]
    idx = cudf.MultiIndex.from_product([idx_1, idx_2])

    data_rand = (
        np.random.default_rng(seed=0)
        .uniform(0, 1, 3 * len(idx))
        .reshape(3, -1)
    )
    df = cudf.DataFrame(data_rand, index=["A", "B", "C"], columns=idx)
    pdf = df.to_pandas()

    assert_eq(df.shape, pdf.shape)

    expected = pdf.loc[["A", "B"]]
    actual = df.loc[["A", "B"]]

    assert_eq(expected, actual)
    assert_eq(df, pdf)


@pytest.mark.parametrize("dtype1", ["int16", "float32"])
@pytest.mark.parametrize("dtype2", ["int16", "float32"])
def test_dataframe_loc_int_float(dtype1, dtype2):
    df = cudf.DataFrame(
        {"a": [10, 11, 12, 13, 14]},
        index=cudf.Index([1, 2, 3, 4, 5], dtype=dtype1),
    )
    pdf = df.to_pandas()

    gidx = cudf.Index([2, 3, 4], dtype=dtype2)
    pidx = gidx.to_pandas()

    actual = df.loc[gidx]
    expected = pdf.loc[pidx]

    assert_eq(actual, expected, check_index_type=True, check_dtype=True)


@pytest.mark.xfail(reason="Not yet properly supported.")
def test_multiindex_wildcard_selection_three_level_all():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2, c3) for c1 in "abcd" for c2 in "abc" for c3 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(24)})
    df.columns = midx

    expect = df.to_pandas().loc[:, (slice("a", "c"), slice("a", "b"), "b")]
    got = df.loc[:, (slice(None), "b")]
    assert_eq(expect, got)


def test_multiindex_wildcard_selection_all():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2) for c1 in "abc" for c2 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(6)})
    df.columns = midx
    expect = df.to_pandas().loc[:, (slice(None), "b")]
    got = df.loc[:, (slice(None), "b")]
    assert_eq(expect, got)


@pytest.mark.xfail(reason="Not yet properly supported.")
def test_multiindex_wildcard_selection_partial():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2) for c1 in "abc" for c2 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(6)})
    df.columns = midx
    expect = df.to_pandas().loc[:, (slice("a", "b"), "b")]
    got = df.loc[:, (slice("a", "b"), "b")]
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "value",
    [
        "7",
        pytest.param(
            ["7", "8"],
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/11298"
            ),
        ),
    ],
)
def test_loc_setitem_string_11298(value):
    df = pd.DataFrame({"a": ["a", "b", "c"]})
    cdf = cudf.from_pandas(df)

    df.loc[:1, "a"] = value

    cdf.loc[:1, "a"] = value

    assert_eq(df, cdf)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/11944")
def test_loc_setitem_list_11944():
    df = pd.DataFrame(
        data={"a": ["yes", "no"], "b": [["l1", "l2"], ["c", "d"]]}
    )
    cdf = cudf.from_pandas(df)
    df.loc[df.a == "yes", "b"] = [["hello"]]
    cdf.loc[cdf.a == "yes", "b"] = [["hello"]]
    assert_eq(df, cdf)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/12504")
def test_loc_setitem_extend_empty_12504():
    df = pd.DataFrame(columns=["a"])
    cdf = cudf.from_pandas(df)

    df.loc[0] = [1]

    cdf.loc[0] = [1]

    assert_eq(df, cdf)


def test_loc_setitem_extend_existing_12505():
    df = pd.DataFrame({"a": [0]})
    cdf = cudf.from_pandas(df)

    df.loc[1] = 1

    cdf.loc[1] = 1

    assert_eq(df, cdf)


def test_loc_setitem_list_arg_missing_raises():
    data = {"a": [0]}
    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    cudf_msg = re.escape("[1] not in the index.")
    with pytest.raises(KeyError, match=cudf_msg):
        gdf.loc[[1]] = 1

    with pytest.raises(KeyError, match=cudf_msg):
        gdf.loc[[1], "a"] = 1

    with pytest.raises(KeyError):
        pdf.loc[[1]] = 1

    with pytest.raises(KeyError):
        pdf.loc[[1], "a"] = 1


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/12801")
def test_loc_setitem_add_column_partial_12801():
    df = pd.DataFrame({"a": [0, 1, 2]})
    cdf = cudf.from_pandas(df)

    df.loc[df.a < 2, "b"] = 1

    cdf.loc[cdf.a < 2, "b"] = 1

    assert_eq(df, cdf)


@contextmanager
def expect_pandas_performance_warning(idx):
    with expect_warning_if(
        (not isinstance(idx[0], tuple) and len(idx) > 2)
        or (isinstance(idx[0], tuple) and len(idx[0]) > 2),
        pd.errors.PerformanceWarning,
    ):
        yield


@pytest.mark.parametrize(
    "key_tuple",
    [
        # return 2 rows, 0 remaining keys = dataframe with entire index
        ("a", "store", "clouds", "fire"),
        (("a", "store", "clouds", "fire"), slice(None)),
        # return 2 rows, 1 remaining key = dataframe with n-k index columns
        ("a", "store", "storm"),
        (("a", "store", "storm"), slice(None)),
        # return 2 rows, 2 remaining keys = dataframe with n-k index columns
        ("a", "store"),
        (("a", "store"), slice(None)),
        # return 2 rows, n-1 remaining keys = dataframe with n-k index columns
        ("a",),
        "a",
        "b",
        "c",
        (("a",), slice(None)),
        # return 1 row, 0 remaining keys = dataframe with entire index
        ("a", "store", "storm", "smoke"),
        (("a", "store", "storm", "smoke"), slice(None)),
        # return 1 row and 1 remaining key = series
        ("c", "forest", "clear"),
        (("c", "forest", "clear"), slice(None)),
    ],
)
def test_multiindex_loc(key_tuple):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    gdf = cudf.from_pandas(pdf)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with expect_pandas_performance_warning(key_tuple):
        expected = pdf.loc[key_tuple]
    got = gdf.loc[key_tuple].sort_index()
    assert_eq(expected.sort_index(), got)

    with cudf.option_context("mode.pandas_compatible", True):
        got = gdf.loc[key_tuple]
    assert_eq(expected, got)


@pytest.mark.parametrize("second_val", [[0, 1], [1, 0]])
def test_multiindex_compatible_ordering(second_val):
    indexer = (([1, 1], second_val), slice(None))
    df = pd.DataFrame(
        {"a": [1, 1, 2, 3], "b": [1, 0, 1, 1], "c": [1, 2, 3, 4]}
    ).set_index(["a", "b"])
    cdf = cudf.from_pandas(df)
    expect = df.loc[indexer]
    with cudf.option_context("mode.pandas_compatible", True):
        actual = cdf.loc[indexer]
    assert_eq(actual, expect)


@pytest.mark.parametrize(
    "arg",
    [
        slice(("a", "store"), ("b", "house")),
        slice(None, ("b", "house")),
        slice(("a", "store"), None),
        slice(None),
    ],
)
def test_multiindex_loc_slice(arg):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdf = cudf.from_pandas(pdf)
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf = pdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[arg], gdf.loc[arg])


def test_multiindex_loc_errors():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    gdf = cudf.from_pandas(pdf)
    gdf.index = gdfIndex

    with pytest.raises(KeyError):
        gdf.loc[("a", "store", "clouds", "foo")]
    with pytest.raises(IndexError):
        gdf.loc[
            ("a", "store", "clouds", "fire", "x", "y")
        ]  # too many indexers
    with pytest.raises(IndexError):
        gdf.loc[slice(None, ("a", "store", "clouds", "fire", "x", "y"))]


def test_multiindex_loc_then_column():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdf = cudf.from_pandas(pdf)
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with pytest.warns(pd.errors.PerformanceWarning):
        expected = pdf.loc[("a", "store", "clouds", "fire"), :][0]
    got = gdf.loc[("a", "store", "clouds", "fire"), :][0]
    assert_eq(expected, got)


def test_multiindex_loc_rows_0():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    gdf = cudf.from_pandas(pdf)
    pdf.index = pdfIndex
    gdf.index = gdfIndex

    assert_exceptions_equal(
        lfunc=pdf.loc.__getitem__,
        rfunc=gdf.loc.__getitem__,
        lfunc_args_and_kwargs=([(("d",), slice(None, None, None))],),
        rfunc_args_and_kwargs=([(("d",), slice(None, None, None))],),
    )


def test_multiindex_loc_rows_1_2_key():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    gdf = cudf.from_pandas(pdf)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[("c", "forest"), :], gdf.loc[("c", "forest"), :])


def test_multiindex_loc_rows_1_1_key():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    gdf = cudf.from_pandas(pdf)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[("c",), :], gdf.loc[("c",), :])


def test_multiindex_index_single_row():
    arrays = [["a", "a", "b", "b"], [1, 2, 3, 4]]
    tuples = list(zip(*arrays, strict=True))
    idx = cudf.MultiIndex.from_tuples(tuples)
    gdf = cudf.DataFrame({"first": range(4), "second": range(4)})
    gdf.index = idx
    pdf = gdf.to_pandas()
    assert_eq(pdf.loc[("b", 3)], gdf.loc[("b", 3)])


@pytest.mark.parametrize("idx_get", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("cols_get", [0, 1, [0, 1], [1, 0], [1], [0]])
def test_multiindex_loc_scalar(idx_get, cols_get):
    idx = cudf.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])
    df = cudf.DataFrame({0: range(4), 1: range(10, 50, 10)}, index=idx)
    pdf = df.to_pandas()

    actual = df.loc[idx_get, cols_get]
    expected = pdf.loc[idx_get, cols_get]

    assert_eq(actual, expected)


def test_multiindex_rows_with_wildcard():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with pytest.warns(pd.errors.PerformanceWarning):
        assert_eq(
            pdf.loc[("a",), :].sort_index(), gdf.loc[("a",), :].sort_index()
        )
        assert_eq(
            pdf.loc[(("a"), ("store")), :].sort_index(),
            gdf.loc[(("a"), ("store")), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(("a"), ("store"), ("storm")), :].sort_index(),
            gdf.loc[(("a"), ("store"), ("storm")), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(("a"), ("store"), ("storm"), ("smoke")), :].sort_index(),
            gdf.loc[(("a"), ("store"), ("storm"), ("smoke")), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(slice(None), "store"), :].sort_index(),
            gdf.loc[(slice(None), "store"), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(slice(None), slice(None), "storm"), :].sort_index(),
            gdf.loc[(slice(None), slice(None), "storm"), :].sort_index(),
        )
        assert_eq(
            pdf.loc[
                (slice(None), slice(None), slice(None), "smoke"), :
            ].sort_index(),
            gdf.loc[
                (slice(None), slice(None), slice(None), "smoke"), :
            ].sort_index(),
        )


def test_multicolumn_loc():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    pdf = pdf.T
    pdf.columns = pdfIndex
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.loc[:, "a"], gdf.loc[:, "a"])
    assert_eq(pdf.loc[:, ("a", "store")], gdf.loc[:, ("a", "store")])
    assert_eq(pdf.loc[:, "a":"b"], gdf.loc[:, "a":"b"])
    assert_eq(pdf.loc[:, ["a", "b"]], gdf.loc[:, ["a", "b"]])

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import DataFrame, Series
from cudf.tests import utils
from cudf.tests.utils import assert_eq

index_dtypes = [np.int64, np.int32, np.int16, np.int8]


@pytest.fixture
def pdf_gdf():
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["c", "d", "e"]}, index=["one", "two", "three"]
    )
    gdf = cudf.from_pandas(pdf)
    return pdf, gdf


@pytest.fixture
def pdf_gdf_multi():
    pdf = pd.DataFrame(np.random.rand(7, 5))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign"]
    pdf.index = pdfIndex
    gdf = cudf.from_pandas(pdf)
    return pdf, gdf


@pytest.mark.parametrize(
    "i1, i2, i3",
    (
        [
            (slice(None, 12), slice(3, None), slice(None, None, 2)),
            (range(12), range(3, 12), range(0, 9, 2)),
            (np.arange(12), np.arange(3, 12), np.arange(0, 9, 2)),
            (list(range(12)), list(range(3, 12)), list(range(0, 9, 2))),
            (
                pd.Series(range(12)),
                pd.Series(range(3, 12)),
                pd.Series(range(0, 9, 2)),
            ),
            (Series(range(12)), Series(range(3, 12)), Series(range(0, 9, 2))),
            (
                [i in range(12) for i in range(20)],
                [i in range(3, 12) for i in range(12)],
                [i in range(0, 9, 2) for i in range(9)],
            ),
            (
                np.array([i in range(12) for i in range(20)], dtype=bool),
                np.array([i in range(3, 12) for i in range(12)], dtype=bool),
                np.array([i in range(0, 9, 2) for i in range(9)], dtype=bool),
            ),
        ]
        + [
            (
                np.arange(12, dtype=t),
                np.arange(3, 12, dtype=t),
                np.arange(0, 9, 2, dtype=t),
            )
            for t in index_dtypes
        ]
    ),
    ids=(
        [
            "slice",
            "range",
            "numpy.array",
            "list",
            "pandas.Series",
            "Series",
            "list[bool]",
            "numpy.array[bool]",
        ]
        + ["numpy.array[%s]" % t.__name__ for t in index_dtypes]
    ),
)
def test_series_indexing(i1, i2, i3):
    a1 = np.arange(20)
    series = Series(a1)
    # Indexing
    sr1 = series[i1]
    assert sr1.null_count == 0
    np.testing.assert_equal(sr1.to_array(), a1[:12])
    sr2 = sr1[i2]
    assert sr2.null_count == 0
    np.testing.assert_equal(sr2.to_array(), a1[3:12])
    # Index with stride
    sr3 = sr2[i3]
    assert sr3.null_count == 0
    np.testing.assert_equal(sr3.to_array(), a1[3:12:2])

    # Integer indexing
    if isinstance(i1, range):
        for i in i1:  # Python int-s
            assert series[i] == a1[i]
    if isinstance(i1, np.ndarray) and i1.dtype in index_dtypes:
        for i in i1:  # numpy integers
            assert series[i] == a1[i]


def test_dataframe_column_name_indexing():
    df = DataFrame()
    data = np.asarray(range(10), dtype=np.int32)
    df["a"] = data
    df[1] = data
    np.testing.assert_equal(
        df["a"].to_array(), np.asarray(range(10), dtype=np.int32)
    )
    np.testing.assert_equal(
        df[1].to_array(), np.asarray(range(10), dtype=np.int32)
    )

    pdf = pd.DataFrame()
    nelem = 10
    pdf["key1"] = np.random.randint(0, 5, nelem)
    pdf["key2"] = np.random.randint(0, 3, nelem)
    pdf[1] = np.arange(1, 1 + nelem)
    pdf[2] = np.random.random(nelem)
    df = DataFrame.from_pandas(pdf)

    assert_eq(df[df.columns], df)
    assert_eq(df[df.columns[:1]], df[["key1"]])

    for i in range(1, len(pdf.columns) + 1):
        for idx in combinations(pdf.columns, i):
            assert pdf[list(idx)].equals(df[list(idx)].to_pandas())

    # test for only numeric columns
    df = pd.DataFrame()
    for i in range(0, 10):
        df[i] = range(nelem)
    gdf = DataFrame.from_pandas(df)
    assert_eq(gdf, df)

    assert_eq(gdf[gdf.columns], gdf)
    assert_eq(gdf[gdf.columns[:3]], gdf[[0, 1, 2]])


def test_dataframe_slicing():
    df = DataFrame()
    size = 123
    df["a"] = ha = np.random.randint(low=0, high=100, size=size).astype(
        np.int32
    )
    df["b"] = hb = np.random.random(size).astype(np.float32)
    df["c"] = hc = np.random.randint(low=0, high=100, size=size).astype(
        np.int64
    )
    df["d"] = hd = np.random.random(size).astype(np.float64)

    # Row slice first 10
    first_10 = df[:10]
    assert len(first_10) == 10
    assert tuple(first_10.columns) == ("a", "b", "c", "d")
    np.testing.assert_equal(first_10["a"].to_array(), ha[:10])
    np.testing.assert_equal(first_10["b"].to_array(), hb[:10])
    np.testing.assert_equal(first_10["c"].to_array(), hc[:10])
    np.testing.assert_equal(first_10["d"].to_array(), hd[:10])
    del first_10

    # Row slice last 10
    last_10 = df[-10:]
    assert len(last_10) == 10
    assert tuple(last_10.columns) == ("a", "b", "c", "d")
    np.testing.assert_equal(last_10["a"].to_array(), ha[-10:])
    np.testing.assert_equal(last_10["b"].to_array(), hb[-10:])
    np.testing.assert_equal(last_10["c"].to_array(), hc[-10:])
    np.testing.assert_equal(last_10["d"].to_array(), hd[-10:])
    del last_10

    # Row slice [begin:end]
    begin = 7
    end = 121
    subrange = df[begin:end]
    assert len(subrange) == end - begin
    assert tuple(subrange.columns) == ("a", "b", "c", "d")
    np.testing.assert_equal(subrange["a"].to_array(), ha[begin:end])
    np.testing.assert_equal(subrange["b"].to_array(), hb[begin:end])
    np.testing.assert_equal(subrange["c"].to_array(), hc[begin:end])
    np.testing.assert_equal(subrange["d"].to_array(), hd[begin:end])
    del subrange


@pytest.mark.parametrize("step", [1, 2, 5])
@pytest.mark.parametrize("scalar", [0, 20, 100])
def test_dataframe_loc(scalar, step):
    size = 123
    pdf = pd.DataFrame(
        {
            "a": np.random.randint(low=0, high=100, size=size),
            "b": np.random.random(size).astype(np.float32),
            "c": np.random.random(size).astype(np.float64),
            "d": np.random.random(size).astype(np.float64),
        }
    )

    df = DataFrame.from_pandas(pdf)

    # Scalar label
    assert_eq(df.loc[scalar], pdf.loc[scalar])

    # Full slice
    assert_eq(df.loc[:, "c"], pdf.loc[:, "c"])

    begin = 110
    end = 122

    assert_eq(
        df.loc[begin:end:step, ["c", "d", "a"]],
        pdf.loc[begin:end:step, ["c", "d", "a"]],
    )

    assert_eq(df.loc[begin:end, ["c", "d"]], pdf.loc[begin:end, ["c", "d"]])

    # Slicing on columns:
    assert_eq(
        df.loc[begin:end:step, "a":"c"], pdf.loc[begin:end:step, "a":"c"]
    )

    # Slicing of size 1:
    assert_eq(df.loc[begin:begin, "a"], pdf.loc[begin:begin, "a"])

    # TODO: Pandas changes the dtype here when it shouldn't
    assert_eq(
        df.loc[begin, "a":"a"], pdf.loc[begin, "a":"a"], check_dtype=False
    )

    # Make int64 index
    offset = 50
    df2 = df[offset:]
    pdf2 = pdf[offset:]
    begin = 117
    end = 122
    assert_eq(
        df2.loc[begin:end, ["c", "d", "a"]],
        pdf2.loc[begin:end, ["c", "d", "a"]],
    )


@pytest.mark.xfail(raises=IndexError, reason="label scalar is out of bound")
def test_dataframe_loc_outbound():
    df = DataFrame()
    size = 10
    df["a"] = ha = np.random.randint(low=0, high=100, size=size).astype(
        np.int32
    )
    df["b"] = hb = np.random.random(size).astype(np.float32)

    pdf = pd.DataFrame()
    pdf["a"] = ha
    pdf["b"] = hb

    np.testing.assert_equal(df.loc[11].to_array(), pdf.loc[11])


def test_series_loc_numerical():
    ps = pd.Series([1, 2, 3, 4, 5], index=[5, 6, 7, 8, 9])
    gs = Series.from_pandas(ps)

    assert_eq(ps.loc[5], gs.loc[5])
    assert_eq(ps.loc[6], gs.loc[6])
    assert_eq(ps.loc[6:8], gs.loc[6:8])
    assert_eq(ps.loc[:8], gs.loc[:8])
    assert_eq(ps.loc[6:], gs.loc[6:])
    assert_eq(ps.loc[::2], gs.loc[::2])
    assert_eq(ps.loc[[5, 8, 9]], gs.loc[[5, 8, 9]])
    assert_eq(
        ps.loc[[True, False, True, False, True]],
        gs.loc[[True, False, True, False, True]],
    )


def test_series_loc_string():
    ps = pd.Series(
        [1, 2, 3, 4, 5], index=["one", "two", "three", "four", "five"]
    )
    gs = Series.from_pandas(ps)

    assert_eq(ps.loc["one"], gs.loc["one"])
    assert_eq(ps.loc["five"], gs.loc["five"])
    assert_eq(ps.loc["two":"four"], gs.loc["two":"four"])
    assert_eq(ps.loc[:"four"], gs.loc[:"four"])
    assert_eq(ps.loc["two":], gs.loc["two":])
    assert_eq(ps.loc[::2], gs.loc[::2])
    assert_eq(ps.loc[["one", "four", "five"]], gs.loc[["one", "four", "five"]])
    assert_eq(
        ps.loc[[True, False, True, False, True]],
        gs.loc[[True, False, True, False, True]],
    )


def test_series_loc_datetime():
    ps = pd.Series(
        [1, 2, 3, 4, 5], index=pd.date_range("20010101", "20010105")
    )
    gs = Series.from_pandas(ps)

    # a few different ways of specifying a datetime label:
    assert_eq(ps.loc["20010101"], gs.loc["20010101"])
    assert_eq(ps.loc["2001-01-01"], gs.loc["2001-01-01"])
    assert_eq(
        ps.loc[pd.to_datetime("2001-01-01")],
        gs.loc[pd.to_datetime("2001-01-01")],
    )
    assert_eq(
        ps.loc[np.datetime64("2001-01-01")],
        gs.loc[np.datetime64("2001-01-01")],
    )

    assert_eq(
        ps.loc["2001-01-02":"2001-01-05"], gs.loc["2001-01-02":"2001-01-05"]
    )
    assert_eq(ps.loc["2001-01-02":], gs.loc["2001-01-02":])
    assert_eq(ps.loc[:"2001-01-04"], gs.loc[:"2001-01-04"])
    assert_eq(ps.loc[::2], gs.loc[::2])
    #
    # assert_eq(ps.loc[['2001-01-01', '2001-01-04', '2001-01-05']],
    #           gs.loc[['2001-01-01', '2001-01-04', '2001-01-05']])
    # looks like a bug in Pandas doesn't let us check for the above,
    # so instead:
    assert_eq(
        ps.loc[
            [
                pd.to_datetime("2001-01-01"),
                pd.to_datetime("2001-01-04"),
                pd.to_datetime("2001-01-05"),
            ]
        ],
        gs.loc[
            [
                pd.to_datetime("2001-01-01"),
                pd.to_datetime("2001-01-04"),
                pd.to_datetime("2001-01-05"),
            ]
        ],
    )
    assert_eq(
        ps.loc[[True, False, True, False, True]],
        gs.loc[[True, False, True, False, True]],
    )


def test_series_loc_categorical():
    ps = pd.Series(
        [1, 2, 3, 4, 5], index=pd.Categorical(["a", "b", "c", "d", "e"])
    )
    gs = Series.from_pandas(ps)

    assert_eq(ps.loc["a"], gs.loc["a"])
    assert_eq(ps.loc["e"], gs.loc["e"])
    assert_eq(ps.loc["b":"d"], gs.loc["b":"d"])
    assert_eq(ps.loc[:"d"], gs.loc[:"d"])
    assert_eq(ps.loc["b":], gs.loc["b":])
    assert_eq(ps.loc[::2], gs.loc[::2])

    # order of categories changes, so we can only
    # compare values:
    assert_eq(
        ps.loc[["a", "d", "e"]].values, gs.loc[["a", "d", "e"]].to_array()
    )

    assert_eq(
        ps.loc[[True, False, True, False, True]],
        gs.loc[[True, False, True, False, True]],
    )


@pytest.mark.parametrize("nelem", [2, 5, 20, 100])
def test_series_iloc(nelem):

    # create random series
    np.random.seed(12)
    ps = pd.Series(np.random.sample(nelem))

    # gpu series
    gs = Series(ps)

    # positive tests for indexing
    np.testing.assert_allclose(gs.iloc[-1 * nelem], ps.iloc[-1 * nelem])
    np.testing.assert_allclose(gs.iloc[-1], ps.iloc[-1])
    np.testing.assert_allclose(gs.iloc[0], ps.iloc[0])
    np.testing.assert_allclose(gs.iloc[1], ps.iloc[1])
    np.testing.assert_allclose(gs.iloc[nelem - 1], ps.iloc[nelem - 1])

    # positive tests for slice
    np.testing.assert_allclose(gs.iloc[-1:1], ps.iloc[-1:1])
    np.testing.assert_allclose(
        gs.iloc[nelem - 1 : -1], ps.iloc[nelem - 1 : -1]
    )
    np.testing.assert_allclose(gs.iloc[0 : nelem - 1], ps.iloc[0 : nelem - 1])
    np.testing.assert_allclose(gs.iloc[0:nelem], ps.iloc[0:nelem])
    np.testing.assert_allclose(gs.iloc[1:1], ps.iloc[1:1])
    np.testing.assert_allclose(gs.iloc[1:2], ps.iloc[1:2])
    np.testing.assert_allclose(
        gs.iloc[nelem - 1 : nelem + 1], ps.iloc[nelem - 1 : nelem + 1]
    )
    np.testing.assert_allclose(
        gs.iloc[nelem : nelem * 2], ps.iloc[nelem : nelem * 2]
    )


@pytest.mark.parametrize("nelem", [2, 5, 20, 100])
def test_dataframe_iloc(nelem):
    gdf = DataFrame()

    gdf["a"] = ha = np.random.randint(low=0, high=100, size=nelem).astype(
        np.int32
    )
    gdf["b"] = hb = np.random.random(nelem).astype(np.float32)

    pdf = pd.DataFrame()
    pdf["a"] = ha
    pdf["b"] = hb

    assert_eq(gdf.iloc[-1:1], pdf.iloc[-1:1])
    assert_eq(gdf.iloc[nelem - 1 : -1], pdf.iloc[nelem - 1 : -1])
    assert_eq(gdf.iloc[0 : nelem - 1], pdf.iloc[0 : nelem - 1])
    assert_eq(gdf.iloc[0:nelem], pdf.iloc[0:nelem])
    assert_eq(gdf.iloc[1:1], pdf.iloc[1:1])
    assert_eq(gdf.iloc[1:2], pdf.iloc[1:2])
    assert_eq(gdf.iloc[nelem - 1 : nelem + 1], pdf.iloc[nelem - 1 : nelem + 1])
    assert_eq(gdf.iloc[nelem : nelem * 2], pdf.iloc[nelem : nelem * 2])

    assert_eq(gdf.iloc[-1 * nelem], pdf.iloc[-1 * nelem])
    assert_eq(gdf.iloc[-1], pdf.iloc[-1])
    assert_eq(gdf.iloc[0], pdf.iloc[0])
    assert_eq(gdf.iloc[1], pdf.iloc[1])
    assert_eq(gdf.iloc[nelem - 1], pdf.iloc[nelem - 1])


@pytest.mark.xfail(raises=AssertionError, reason="Series.index are different")
def test_dataframe_iloc_tuple():
    gdf = DataFrame()
    nelem = 123
    gdf["a"] = ha = np.random.randint(low=0, high=100, size=nelem).astype(
        np.int32
    )
    gdf["b"] = hb = np.random.random(nelem).astype(np.float32)

    pdf = pd.DataFrame()
    pdf["a"] = ha
    pdf["b"] = hb

    # We don't support passing the column names into the index quite yet
    got = gdf.iloc[1, [1]]
    expect = pdf.iloc[1, [1]]

    assert_eq(got, expect)


@pytest.mark.xfail(
    raises=IndexError, reason="positional indexers are out-of-bounds"
)
def test_dataframe_iloc_index_error():
    gdf = DataFrame()
    nelem = 123
    gdf["a"] = ha = np.random.randint(low=0, high=100, size=nelem).astype(
        np.int32
    )
    gdf["b"] = hb = np.random.random(nelem).astype(np.float32)

    pdf = pd.DataFrame()
    pdf["a"] = ha
    pdf["b"] = hb

    def assert_col(g, p):
        np.testing.assert_equal(g["a"].to_array(), p["a"])
        np.testing.assert_equal(g["b"].to_array(), p["b"])

    assert_col(gdf.iloc[nelem * 2], pdf.iloc[nelem * 2])


@pytest.mark.parametrize("ntake", [0, 1, 10, 123, 122, 200])
def test_dataframe_take(ntake):
    np.random.seed(0)
    df = DataFrame()

    nelem = 123
    df["ii"] = ii = np.random.randint(0, 20, nelem)
    df["ff"] = ff = np.random.random(nelem)

    take_indices = np.random.randint(0, len(df), ntake)

    def check(**kwargs):
        out = df.take(take_indices, **kwargs)
        assert len(out) == ntake
        assert out.ii.null_count == 0
        assert out.ff.null_count == 0
        np.testing.assert_array_equal(out.ii.to_array(), ii[take_indices])
        np.testing.assert_array_equal(out.ff.to_array(), ff[take_indices])
        if kwargs.get("ignore_index"):
            np.testing.assert_array_equal(out.index, np.arange(ntake))
        else:
            np.testing.assert_array_equal(out.index, take_indices)

    check()
    check(ignore_index=True)


@pytest.mark.parametrize("nelem", [0, 1, 5, 20, 100])
@pytest.mark.parametrize("slice_start", [None, 0, 1, 3, 10, -10])
@pytest.mark.parametrize("slice_end", [None, 0, 1, 30, 50, -1])
def test_dataframe_masked_slicing(nelem, slice_start, slice_end):
    gdf = DataFrame()
    gdf["a"] = list(range(nelem))
    gdf["b"] = list(range(nelem, 2 * nelem))
    gdf["a"] = gdf["a"].set_mask(utils.random_bitmask(nelem))
    gdf["b"] = gdf["b"].set_mask(utils.random_bitmask(nelem))

    def do_slice(x):
        return x[slice_start:slice_end]

    expect = do_slice(gdf.to_pandas())
    got = do_slice(gdf).to_pandas()

    pd.testing.assert_frame_equal(expect, got)


def test_dataframe_boolean_mask_with_None():
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
    gdf = DataFrame.from_pandas(pdf)
    pdf_masked = pdf[[True, False, True, False]]
    gdf_masked = gdf[[True, False, True, False]]
    assert_eq(pdf_masked, gdf_masked)


@pytest.mark.parametrize("dtype", [int, float, str])
def test_empty_boolean_mask(dtype):
    gdf = cudf.datasets.randomdata(nrows=0, dtypes={"a": dtype})
    pdf = gdf.to_pandas()

    expected = pdf[pdf.a == 1]
    got = gdf[gdf.a == 1]
    assert_eq(expected, got)

    expected = pdf.a[pdf.a == 1]
    got = gdf.a[gdf.a == 1]
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4],
        [1.0, 2.0, 3.0, 4.0],
        ["one", "two", "three", "four"],
        pd.Series(["a", "b", "c", "d"], dtype="category"),
        pd.Series(pd.date_range("2010-01-01", "2010-01-04")),
    ],
)
@pytest.mark.parametrize(
    "mask",
    [
        [True, True, True, True],
        [False, False, False, False],
        [True, False, True, False],
        [True, False, False, True],
        np.array([True, False, True, False]),
        pd.Series([True, False, True, False]),
        cudf.Series([True, False, True, False]),
    ],
)
@pytest.mark.parametrize("nulls", ["one", "some", "all", "none"])
def test_series_apply_boolean_mask(data, mask, nulls):
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

    # TODO: from_pandas(psr) has dtype "float64"
    # when psr has dtype "object" and is all None
    if psr.dtype == "object" and nulls == "all":
        gsr = cudf.Series([None, None, None, None], dtype="object")

    if isinstance(mask, cudf.Series):
        expect = psr[mask.to_pandas()]
    else:
        expect = psr[mask]
    got = gsr[mask]

    assert_eq(expect, got)


def test_dataframe_apply_boolean_mask():
    pdf = pd.DataFrame(
        {
            "a": [0, 1, 2, 3],
            "b": [0.1, 0.2, None, 0.3],
            "c": ["a", None, "b", "c"],
        }
    )
    gdf = DataFrame.from_pandas(pdf)
    assert_eq(pdf[[True, False, True, False]], gdf[[True, False, True, False]])


"""
This test compares cudf and Pandas dataframe boolean indexing.
"""


@pytest.mark.parametrize(
    "mask_fn", [lambda x: x, lambda x: np.array(x), lambda x: pd.Series(x)]
)
def test_dataframe_boolean_mask(mask_fn):
    mask_base = [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.from_pandas(pdf)
    mask = mask_fn(mask_base)
    assert len(mask) == gdf.shape[0]
    pdf_masked = pdf[mask]
    gdf_masked = gdf[mask]
    assert pdf_masked.to_string().split() == gdf_masked.to_string().split()


@pytest.mark.parametrize(
    "key, value",
    [
        (0, 4),
        (1, 4),
        ([0, 1], 4),
        ([0, 1], [4, 5]),
        (slice(0, 2), [4, 5]),
        (slice(1, None), [4, 5, 6, 7]),
        ([], 1),
        ([], []),
        (slice(None, None), 1),
        (slice(-1, -3), 7),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some", "all"])
def test_series_setitem_basics(key, value, nulls):
    psr = pd.Series([1, 2, 3, 4, 5])
    if nulls == "some":
        psr[[0, 4]] = None
    elif nulls == "all":
        psr[:] = None
    gsr = cudf.from_pandas(psr)
    psr[key] = value
    gsr[key] = value
    assert_eq(psr, gsr, check_dtype=False)


def test_series_setitem_null():
    gsr = cudf.Series([1, 2, 3, 4])
    gsr[0] = None

    expect = cudf.Series([None, 2, 3, 4])
    got = gsr
    assert_eq(expect, got)

    gsr = cudf.Series([None, 2, 3, 4])
    gsr[0] = 1

    expect = cudf.Series([1, 2, 3, 4])
    got = gsr
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "key, value",
    [
        (0, 4),
        (1, 4),
        ([0, 1], 4),
        ([0, 1], [4, 5]),
        (slice(0, 2), [4, 5]),
        (slice(1, None), [4, 5, 6, 7]),
        ([], 1),
        ([], []),
        (slice(None, None), 1),
        (slice(-1, -3), 7),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some", "all"])
def test_series_setitem_iloc(key, value, nulls):
    psr = pd.Series([1, 2, 3, 4, 5])
    if nulls == "some":
        psr[[0, 4]] = None
    elif nulls == "all":
        psr[:] = None
    gsr = cudf.from_pandas(psr)
    psr.iloc[key] = value
    gsr.iloc[key] = value
    assert_eq(psr, gsr, check_dtype=False)


@pytest.mark.parametrize(
    "key, value",
    [
        (0, 0.5),
        ([0, 1], 0.5),
        ([0, 1], [0.5, 2.5]),
        (slice(0, 2), [0.5, 0.25]),
    ],
)
def test_series_setitem_dtype(key, value):
    psr = pd.Series([1, 2, 3], dtype="int32")
    gsr = cudf.from_pandas(psr)
    psr[key] = value
    gsr[key] = value
    assert_eq(psr, gsr)


def test_series_setitem_datetime():
    psr = pd.Series(["2001", "2002", "2003"], dtype="datetime64[ns]")
    gsr = cudf.from_pandas(psr)

    psr[0] = "2005"
    gsr[0] = "2005"

    assert_eq(psr, gsr)


def test_series_setitem_categorical():
    psr = pd.Series(["a", "b", "a", "c", "d"], dtype="category")
    gsr = cudf.from_pandas(psr)

    psr[0] = "d"
    gsr[0] = "d"
    assert_eq(psr, gsr)

    psr = psr.cat.add_categories(["e"])
    gsr = gsr.cat.add_categories(["e"])
    psr[0] = "e"
    gsr[0] = "e"
    assert_eq(psr, gsr)

    psr[[0, 1]] = "b"
    gsr[[0, 1]] = "b"
    assert_eq(psr, gsr)

    psr[0:3] = "e"
    gsr[0:3] = "e"
    assert_eq(psr, gsr)


@pytest.mark.parametrize(
    "key, value",
    [
        (0, "d"),
        (0, "g"),
        ([0, 1], "g"),
        ([0, 1], None),
        (slice(None, 2), "g"),
        (slice(None, 2), ["g", None]),
    ],
)
def test_series_setitem_string(key, value):
    psr = pd.Series(["a", "b", "c", "d", "e"])
    gsr = cudf.from_pandas(psr)
    psr[key] = value
    gsr[key] = value
    assert_eq(psr, gsr)

    psr = pd.Series(["a", None, "c", "d", "e"])
    gsr = cudf.from_pandas(psr)
    psr[key] = value
    gsr[key] = value
    assert_eq(psr, gsr)


@pytest.mark.parametrize(
    "key, value",
    [
        ("a", 4),
        ("b", 4),
        (["a", "b"], 4),
        (["a", "b"], [4, 5]),
        ([True, False, True], 4),
        ([False, False, False], 4),
        ([True, False, True], [4, 5]),
    ],
)
def test_series_setitem_loc(key, value):
    psr = pd.Series([1, 2, 3], ["a", "b", "c"])
    gsr = cudf.from_pandas(psr)
    psr.loc[key] = value
    gsr.loc[key] = value
    assert_eq(psr, gsr)


@pytest.mark.parametrize(
    "key, value",
    [
        ((0, 0), 5),
        ((slice(None), 0), 5),
        ((slice(None), 0), range(3)),
        ((slice(None, -1), 0), range(2)),
        (([0, 1], 0), 5),
    ],
)
def test_dataframe_setitem_iloc(key, value, pdf_gdf):
    pdf, gdf = pdf_gdf
    pdf.iloc[key] = value
    gdf.iloc[key] = value
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "key, value",
    [
        (("one", "a"), 5),
        ((slice(None), "a"), 5),
        ((slice(None), "a"), range(3)),
        ((slice(None, "two"), "a"), range(2)),
        ((["one", "two"], "a"), 5),
    ],
)
def test_dataframe_setitem_loc(key, value, pdf_gdf):
    pdf, gdf = pdf_gdf
    pdf.loc[key] = value
    gdf.loc[key] = value
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "key,value",
    [
        ((0, 0), 5.0),
        ((slice(None), 0), 5.0),
        ((slice(None), 0), np.arange(7, dtype="float64")),
    ],
)
def test_dataframe_setitem_iloc_multiindex(key, value, pdf_gdf_multi):
    pdf, gdf = pdf_gdf_multi

    pdf.iloc[key] = value
    gdf.iloc[key] = value

    assert_eq(pdf, gdf)

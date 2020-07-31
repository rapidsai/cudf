from itertools import combinations

import cupy
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import DataFrame, Series
from cudf.tests import utils
from cudf.tests.utils import INTEGER_TYPES, assert_eq

index_dtypes = INTEGER_TYPES


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
        + ["numpy.array[%s]" % np.dtype(t).type.__name__ for t in index_dtypes]
    ),
)
def test_series_indexing(i1, i2, i3):
    a1 = np.arange(20)
    series = Series(a1)
    # Indexing
    sr1 = series.iloc[i1]
    assert sr1.null_count == 0
    np.testing.assert_equal(sr1.to_array(), a1[:12])
    sr2 = sr1.iloc[i2]
    assert sr2.null_count == 0
    np.testing.assert_equal(sr2.to_array(), a1[3:12])
    # Index with stride
    sr3 = sr2.iloc[i3]
    assert sr3.null_count == 0
    np.testing.assert_equal(sr3.to_array(), a1[3:12:2])

    # Integer indexing
    if isinstance(i1, range):
        for i in i1:  # Python int-s
            assert series[i] == a1[i]
    if isinstance(i1, np.ndarray) and i1.dtype in index_dtypes:
        for i in i1:  # numpy integers
            assert series[i] == a1[i]


def test_series_indexing_large_size():
    n_elem = 100_000
    gsr = cudf.Series(cupy.ones(n_elem))
    gsr[0] = None
    got = gsr[gsr.isna()]
    expect = Series([None], dtype="float64")

    assert_eq(expect, got)


@pytest.mark.parametrize("psr", [pd.Series([1, 2, 3], index=["a", "b", "c"])])
@pytest.mark.parametrize(
    "arg", ["b", ["a", "c"], slice(1, 2, 1), [True, False, True]]
)
def test_series_get_item(psr, arg):
    gsr = Series.from_pandas(psr)

    expect = psr[arg]
    got = gsr[arg]

    assert_eq(expect, got)


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
            assert pdf[list(idx)].equals(
                df[list(idx)].to_pandas(nullable_pd_dtype=False)
            )

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

    assert_eq(df.loc[:, ["a"]], pdf.loc[:, ["a"]])

    assert_eq(df.loc[:, "d"], pdf.loc[:, "d"])

    # Scalar label
    assert_eq(df.loc[scalar], pdf.loc[scalar])

    # Full slice
    assert_eq(df.loc[:, "c"], pdf.loc[:, "c"])

    # Repeat with at[]
    assert_eq(df.loc[:, ["a"]], df.at[:, ["a"]])
    assert_eq(df.loc[:, "d"], df.at[:, "d"])
    assert_eq(df.loc[scalar], df.at[scalar])
    assert_eq(df.loc[:, "c"], df.at[:, "c"])

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

    # Repeat with at[]
    assert_eq(
        df.loc[begin:end:step, ["c", "d", "a"]],
        df.at[begin:end:step, ["c", "d", "a"]],
    )
    assert_eq(df.loc[begin:end, ["c", "d"]], df.at[begin:end, ["c", "d"]])
    assert_eq(df.loc[begin:end:step, "a":"c"], df.at[begin:end:step, "a":"c"])
    assert_eq(df.loc[begin:begin, "a"], df.at[begin:begin, "a"])
    assert_eq(df.loc[begin, "a":"a"], df.at[begin, "a":"a"], check_dtype=False)

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

    # loc with list like indexing
    assert_eq(df.loc[[0]], pdf.loc[[0]])
    # loc with column like indexing
    assert_eq(df.loc[cudf.Series([0])], pdf.loc[pd.Series([0])])
    assert_eq(df.loc[cudf.Series([0])._column], pdf.loc[pd.Series([0])])
    assert_eq(df.loc[np.array([0])], pdf.loc[np.array([0])])


def test_dataframe_loc_duplicate_index_scalar():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=[1, 2, 1, 4, 2])
    gdf = DataFrame.from_pandas(pdf)

    assert_eq(pdf.loc[2], gdf.loc[2])


@pytest.mark.parametrize(
    "mask",
    [[True, False, False, False, False], [True, False, True, False, True]],
)
@pytest.mark.parametrize("arg", ["a", slice("a", "a"), slice("a", "b")])
def test_dataframe_loc_mask(mask, arg):
    pdf = pd.DataFrame(
        {"a": ["a", "b", "c", "d", "e"], "b": ["f", "g", "h", "i", "j"]}
    )
    gdf = DataFrame.from_pandas(pdf)

    assert_eq(pdf.loc[mask, arg], gdf.loc[mask, arg])


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
    assert_eq(ps.loc[[5, 8, 9]], gs.loc[cupy.array([5, 8, 9])])


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


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            {"a": [1, 2, 3, 4]},
            index=pd.MultiIndex.from_frame(
                pd.DataFrame(
                    {"A": [2, 3, 1, 4], "B": ["low", "high", "high", "low"]}
                )
            ),
        ),
        pd.Series(
            [1, 2, 3, 4],
            index=pd.MultiIndex.from_frame(
                pd.DataFrame(
                    {"A": [2, 3, 1, 4], "B": ["low", "high", "high", "low"]}
                )
            ),
        ),
    ],
)
def test_dataframe_series_loc_multiindex(obj):
    pindex = pd.MultiIndex.from_frame(
        pd.DataFrame({"A": [3, 2], "B": ["high", "low"]})
    )

    gobj = cudf.from_pandas(obj)
    gindex = cudf.MultiIndex.from_pandas(pindex)

    # cudf MultiIndex as arg
    expected = obj.loc[pindex]
    got = gobj.loc[gindex]
    assert_eq(expected, got)

    # pandas MultiIndex as arg
    expected = obj.loc[pindex]
    got = gobj.loc[pindex]
    assert_eq(expected, got)


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
    np.testing.assert_allclose(gs.iloc[-1:1].to_array(), ps.iloc[-1:1])
    np.testing.assert_allclose(
        gs.iloc[nelem - 1 : -1].to_array(), ps.iloc[nelem - 1 : -1]
    )
    np.testing.assert_allclose(
        gs.iloc[0 : nelem - 1].to_pandas(), ps.iloc[0 : nelem - 1]
    )
    np.testing.assert_allclose(gs.iloc[0:nelem].to_pandas(), ps.iloc[0:nelem])
    np.testing.assert_allclose(gs.iloc[1:1].to_pandas(), ps.iloc[1:1])
    np.testing.assert_allclose(gs.iloc[1:2].to_pandas(), ps.iloc[1:2].values)
    np.testing.assert_allclose(
        gs.iloc[nelem - 1 : nelem + 1].to_pandas(),
        ps.iloc[nelem - 1 : nelem + 1],
    )
    np.testing.assert_allclose(
        gs.iloc[nelem : nelem * 2].to_pandas(), ps.iloc[nelem : nelem * 2]
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

    # Repeat the above with iat[]
    assert_eq(gdf.iloc[-1:1], gdf.iat[-1:1])
    assert_eq(gdf.iloc[nelem - 1 : -1], gdf.iat[nelem - 1 : -1])
    assert_eq(gdf.iloc[0 : nelem - 1], gdf.iat[0 : nelem - 1])
    assert_eq(gdf.iloc[0:nelem], gdf.iat[0:nelem])
    assert_eq(gdf.iloc[1:1], gdf.iat[1:1])
    assert_eq(gdf.iloc[1:2], gdf.iat[1:2])
    assert_eq(gdf.iloc[nelem - 1 : nelem + 1], gdf.iat[nelem - 1 : nelem + 1])
    assert_eq(gdf.iloc[nelem : nelem * 2], gdf.iat[nelem : nelem * 2])

    assert_eq(gdf.iloc[-1 * nelem], gdf.iat[-1 * nelem])
    assert_eq(gdf.iloc[-1], gdf.iat[-1])
    assert_eq(gdf.iloc[0], gdf.iat[0])
    assert_eq(gdf.iloc[1], gdf.iat[1])
    assert_eq(gdf.iloc[nelem - 1], gdf.iat[nelem - 1])

    # iloc with list like indexing
    assert_eq(gdf.iloc[[0]], pdf.iloc[[0]])
    # iloc with column like indexing
    assert_eq(gdf.iloc[cudf.Series([0])], pdf.iloc[pd.Series([0])])
    assert_eq(gdf.iloc[cudf.Series([0])._column], pdf.iloc[pd.Series([0])])
    assert_eq(gdf.iloc[np.array([0])], pdf.loc[np.array([0])])


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

    out = df.take(take_indices)
    assert len(out) == ntake
    assert out.ii.null_count == 0
    assert out.ff.null_count == 0
    np.testing.assert_array_equal(out.ii.to_array(), ii[take_indices])
    np.testing.assert_array_equal(out.ff.to_array(), ff[take_indices])
    np.testing.assert_array_equal(out.index.to_array(), take_indices)


@pytest.mark.parametrize("keep_index", [True, False])
@pytest.mark.parametrize("ntake", [0, 1, 10, 123, 122, 200])
def test_series_take(ntake, keep_index):
    np.random.seed(0)
    nelem = 123

    data = np.random.randint(0, 20, nelem)
    sr = Series(data)

    take_indices = np.random.randint(0, len(sr), ntake)

    if keep_index is True:
        out = sr.take(take_indices)
        np.testing.assert_array_equal(out.to_array(), data[take_indices])
    elif keep_index is False:
        out = sr.take(take_indices, keep_index=False)
        np.testing.assert_array_equal(out.to_array(), data[take_indices])
        np.testing.assert_array_equal(
            out.index.to_array(), sr.index.to_array()
        )


def test_series_take_positional():
    psr = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])

    gsr = Series.from_pandas(psr)

    take_indices = [1, 2, 0, 3]

    expect = psr.take(take_indices)
    got = gsr.take(take_indices, keep_index=True)

    assert_eq(expect, got)


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

    assert_eq(expect, got)


def test_dataframe_boolean_mask_with_None():
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
    gdf = DataFrame.from_pandas(pdf)
    pdf_masked = pdf[[True, False, True, False]]
    gdf_masked = gdf[[True, False, True, False]]
    assert_eq(pdf_masked, gdf_masked)

    with pytest.raises(ValueError):
        gdf[Series([True, False, None, False])]


@pytest.mark.parametrize("dtype", [int, float, str])
def test_empty_boolean_mask(dtype):
    gdf = cudf.datasets.randomdata(nrows=0, dtypes={"a": dtype})
    pdf = gdf.to_pandas(nullable_pd_dtype=False)

    compare_val = dtype(1)

    expected = pdf[pdf.a == compare_val]
    got = gdf[gdf.a == compare_val]
    assert_eq(expected, got)

    expected = pdf.a[pdf.a == compare_val]
    got = gdf.a[gdf.a == compare_val]
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

    psr[0] = np.datetime64("2005")
    gsr[0] = np.datetime64("2005")

    assert_eq(psr, gsr)


@pytest.mark.xfail(reason="Pandas will coerce to object datatype here")
def test_series_setitem_datetime_coerced():
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


def test_boolean_indexing_single_row(pdf_gdf):
    pdf, gdf = pdf_gdf
    assert_eq(
        pdf.loc[[True, False, False], :], gdf.loc[[True, False, False], :]
    )


def test_iloc_negative_indices():
    psr = pd.Series([1, 2, 3, 4, 5])
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.iloc[[-1, -2, -4]], gsr.iloc[[-1, -2, -4]])


def test_out_of_bounds_indexing():
    a = cudf.Series([1, 2, 3])
    with pytest.raises(IndexError):
        a[[0, 1, 9]]
    with pytest.raises(IndexError):
        a[[0, 1, -4]]
    with pytest.raises(IndexError):
        a[[0, 1, 9]] = 2
    with pytest.raises(IndexError):
        a[[0, 1, -4]] = 2
    with pytest.raises(IndexError):
        a[4:6].iloc[-1] = 2
    with pytest.raises(IndexError):
        a[4:6].iloc[1] = 2


def test_sliced_indexing():
    a = list(range(4, 4 + 150))
    b = list(range(0, 0 + 150))
    pdf = pd.DataFrame({"a": a, "b": b})
    gdf = DataFrame.from_pandas(pdf)
    pdf = pdf.set_index("a")
    gdf = gdf.set_index("a")
    pidx = pdf.index[:75]
    gidx = gdf.index[:75]

    assert_eq(pdf.loc[pidx], gdf.loc[gidx])


@pytest.mark.parametrize("index", [["a"], ["a", "a"], ["a", "a", "b", "c"]])
def test_iloc_categorical_index(index):
    gdf = cudf.DataFrame({"data": range(len(index))}, index=index)
    gdf.index = gdf.index.astype("category")
    pdf = gdf.to_pandas(nullable_pd_dtype=False)
    expect = pdf.iloc[:, 0]
    got = gdf.iloc[:, 0]
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "sli",
    [
        slice("2001", "2020"),
        slice("2001", "2002"),
        slice("2002", "2001"),
        slice(None, "2020"),
        slice("2020", None),
    ],
)
@pytest.mark.parametrize("is_dataframe", [True, False])
def test_loc_datetime_index(sli, is_dataframe):

    if is_dataframe is True:
        pd_data = pd.DataFrame(
            {"a": [1, 2, 3]},
            index=pd.Series(["2001", "2009", "2002"], dtype="datetime64[ns]"),
        )
    else:
        pd_data = pd.Series(
            [1, 2, 3],
            pd.Series(["2001", "2009", "2002"], dtype="datetime64[ns]"),
        )

    gd_data = cudf.from_pandas(pd_data)

    expect = pd_data.loc[sli]
    got = gd_data.loc[sli]

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "gdf",
    [
        cudf.DataFrame({"a": range(1000000)}),
        cudf.DataFrame({"a": range(1000000), "b": range(1000000)}),
        cudf.DataFrame({"a": range(20), "b": range(20)}),
        cudf.DataFrame(
            {
                "a": range(20),
                "b": range(20),
                "c": ["abc", "def", "xyz", "def", "pqr"] * 4,
            }
        ),
        cudf.DataFrame(index=[1, 2, 3]),
        cudf.DataFrame(index=range(1000000)),
        cudf.DataFrame(columns=["a", "b", "c", "d"]),
        cudf.DataFrame(columns=["a"], index=range(1000000)),
        cudf.DataFrame(
            columns=["a", "col2", "...col n"], index=range(1000000)
        ),
        cudf.DataFrame(index=cudf.Series(range(1000000)).astype("str")),
        cudf.DataFrame(
            columns=["a", "b", "c", "d"],
            index=cudf.Series(range(1000000)).astype("str"),
        ),
    ],
)
@pytest.mark.parametrize(
    "slice",
    [
        slice(250000, 500000),
        slice(250000, 250001),
        slice(500000),
        slice(1, 10),
        slice(10, 20),
        slice(15, 24000),
        slice(6),
    ],
)
def test_dataframe_sliced(gdf, slice):
    pdf = gdf.to_pandas(nullable_pd_dtype=False)

    actual = gdf[slice]
    expected = pdf[slice]

    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "gdf",
    [
        cudf.DataFrame({"a": range(10000)}),
        cudf.DataFrame(
            {
                "a": range(10000),
                "b": range(10000),
                "c": range(10000),
                "d": range(10000),
                "e": range(10000),
                "f": range(10000),
            }
        ),
        cudf.DataFrame({"a": range(20), "b": range(20)}),
        cudf.DataFrame(
            {
                "a": range(20),
                "b": range(20),
                "c": ["abc", "def", "xyz", "def", "pqr"] * 4,
            }
        ),
        cudf.DataFrame(index=[1, 2, 3]),
        cudf.DataFrame(index=range(10000)),
        cudf.DataFrame(columns=["a", "b", "c", "d"]),
        cudf.DataFrame(columns=["a"], index=range(10000)),
        cudf.DataFrame(columns=["a", "col2", "...col n"], index=range(10000)),
        cudf.DataFrame(index=cudf.Series(range(10000)).astype("str")),
        cudf.DataFrame(
            columns=["a", "b", "c", "d"],
            index=cudf.Series(range(10000)).astype("str"),
        ),
    ],
)
@pytest.mark.parametrize(
    "slice", [slice(6), slice(1), slice(7), slice(1, 3)],
)
def test_dataframe_iloc_index(gdf, slice):
    pdf = gdf.to_pandas(nullable_pd_dtype=False)

    actual = gdf.iloc[:, slice]
    expected = pdf.iloc[:, slice]

    assert_eq(actual, expected)

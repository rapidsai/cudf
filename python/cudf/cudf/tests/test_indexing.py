# Copyright (c) 2021-2024, NVIDIA CORPORATION.

import weakref
from datetime import datetime
from itertools import combinations

import cupy
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.testing import _utils as utils
from cudf.testing._utils import (
    INTEGER_TYPES,
    assert_eq,
    assert_exceptions_equal,
    expect_warning_if,
)

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
            (
                cudf.Series(range(12)),
                cudf.Series(range(3, 12)),
                cudf.Series(range(0, 9, 2)),
            ),
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
    series = cudf.Series(a1)

    # Indexing
    sr1 = series.iloc[i1]
    assert sr1.null_count == 0
    np.testing.assert_equal(sr1.to_numpy(), a1[:12])

    sr2 = sr1.iloc[i2]
    assert sr2.null_count == 0
    np.testing.assert_equal(sr2.to_numpy(), a1[3:12])

    # Index with stride
    sr3 = sr2.iloc[i3]
    assert sr3.null_count == 0
    np.testing.assert_equal(sr3.to_numpy(), a1[3:12:2])

    # Integer indexing
    if isinstance(i1, range):
        for i in i1:  # Python int-s
            assert series[i] == a1[i]
    if isinstance(i1, np.ndarray) and i1.dtype in index_dtypes:
        for i in i1:  # numpy integers
            assert series[i] == a1[i]


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "arg",
    [
        1,
        -1,
        "b",
        np.int32(1),
        np.uint32(1),
        np.int8(1),
        np.uint8(1),
        np.int16(1),
        np.uint16(1),
        np.int64(1),
        np.uint64(1),
    ],
)
def test_series_get_item_iloc_defer(arg):
    # Indexing for non-numeric dtype Index
    ps = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"]))
    gs = cudf.from_pandas(ps)

    arg_not_str = not isinstance(arg, str)
    with expect_warning_if(arg_not_str):
        expect = ps[arg]
    with expect_warning_if(arg_not_str):
        got = gs[arg]

    assert_eq(expect, got)


def test_series_iloc_defer_cudf_scalar():
    ps = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"]))
    gs = cudf.from_pandas(ps)

    for t in index_dtypes:
        arg = cudf.Scalar(1, dtype=t)
        got = gs.iloc[arg]
        expect = 2
        assert_eq(expect, got)


def test_series_indexing_large_size():
    n_elem = 100_000
    gsr = cudf.Series(cupy.ones(n_elem))
    gsr[0] = None
    got = gsr[gsr.isna()]
    expect = cudf.Series([None], dtype="float64")

    assert_eq(expect, got)


@pytest.mark.parametrize("psr", [pd.Series([1, 2, 3], index=["a", "b", "c"])])
@pytest.mark.parametrize(
    "arg", ["b", ["a", "c"], slice(1, 2, 1), [True, False, True]]
)
def test_series_get_item(psr, arg):
    gsr = cudf.from_pandas(psr)

    expect = psr[arg]
    got = gsr[arg]

    assert_eq(expect, got)


def test_dataframe_column_name_indexing():
    df = cudf.DataFrame()
    data = np.asarray(range(10), dtype=np.int32)
    df["a"] = data
    df[1] = data
    np.testing.assert_equal(
        df["a"].to_numpy(), np.asarray(range(10), dtype=np.int32)
    )
    np.testing.assert_equal(
        df[1].to_numpy(), np.asarray(range(10), dtype=np.int32)
    )

    pdf = pd.DataFrame()
    nelem = 10
    pdf["key1"] = np.random.randint(0, 5, nelem)
    pdf["key2"] = np.random.randint(0, 3, nelem)
    pdf[1] = np.arange(1, 1 + nelem)
    pdf[2] = np.random.random(nelem)
    df = cudf.from_pandas(pdf)

    assert_eq(df[df.columns], df)
    assert_eq(df[df.columns[:1]], df[["key1"]])

    for i in range(1, len(pdf.columns) + 1):
        for idx in combinations(pdf.columns, i):
            assert pdf[list(idx)].equals(df[list(idx)].to_pandas())

    # test for only numeric columns
    df = pd.DataFrame()
    for i in range(0, 10):
        df[i] = range(nelem)
    gdf = cudf.DataFrame.from_pandas(df)
    assert_eq(gdf, df)

    assert_eq(gdf[gdf.columns], gdf)
    assert_eq(gdf[gdf.columns[:3]], gdf[[0, 1, 2]])


def test_dataframe_slicing():
    df = cudf.DataFrame()
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
    np.testing.assert_equal(first_10["a"].to_numpy(), ha[:10])
    np.testing.assert_equal(first_10["b"].to_numpy(), hb[:10])
    np.testing.assert_equal(first_10["c"].to_numpy(), hc[:10])
    np.testing.assert_equal(first_10["d"].to_numpy(), hd[:10])
    del first_10

    # Row slice last 10
    last_10 = df[-10:]
    assert len(last_10) == 10
    assert tuple(last_10.columns) == ("a", "b", "c", "d")
    np.testing.assert_equal(last_10["a"].to_numpy(), ha[-10:])
    np.testing.assert_equal(last_10["b"].to_numpy(), hb[-10:])
    np.testing.assert_equal(last_10["c"].to_numpy(), hc[-10:])
    np.testing.assert_equal(last_10["d"].to_numpy(), hd[-10:])
    del last_10

    # Row slice [begin:end]
    begin = 7
    end = 121
    subrange = df[begin:end]
    assert len(subrange) == end - begin
    assert tuple(subrange.columns) == ("a", "b", "c", "d")
    np.testing.assert_equal(subrange["a"].to_numpy(), ha[begin:end])
    np.testing.assert_equal(subrange["b"].to_numpy(), hb[begin:end])
    np.testing.assert_equal(subrange["c"].to_numpy(), hc[begin:end])
    np.testing.assert_equal(subrange["d"].to_numpy(), hd[begin:end])
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
    pdf.index.name = "index"

    df = cudf.DataFrame.from_pandas(pdf)

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
    gdf = cudf.DataFrame.from_pandas(pdf)

    pdf_sorted = pdf.sort_values(by=list(pdf.columns), axis=0)
    gdf_sorted = gdf.sort_values(by=list(gdf.columns), axis=0)

    assert_eq(pdf_sorted, gdf_sorted)


@pytest.mark.parametrize(
    "mask",
    [[True, False, False, False, False], [True, False, True, False, True]],
)
@pytest.mark.parametrize("arg", ["a", slice("a", "a"), slice("a", "b")])
def test_dataframe_loc_mask(mask, arg):
    pdf = pd.DataFrame(
        {"a": ["a", "b", "c", "d", "e"], "b": ["f", "g", "h", "i", "j"]}
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

    assert_eq(pdf.loc[mask, arg], gdf.loc[mask, arg])


def test_dataframe_loc_outbound():
    df = cudf.DataFrame()
    size = 10
    df["a"] = ha = np.random.randint(low=0, high=100, size=size).astype(
        np.int32
    )
    df["b"] = hb = np.random.random(size).astype(np.float32)

    pdf = pd.DataFrame()
    pdf["a"] = ha
    pdf["b"] = hb

    assert_exceptions_equal(lambda: pdf.loc[11], lambda: df.loc[11])


def test_series_loc_numerical():
    ps = pd.Series([1, 2, 3, 4, 5], index=[5, 6, 7, 8, 9])
    gs = cudf.Series.from_pandas(ps)

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


def test_series_loc_float_index():
    ps = pd.Series([1, 2, 3, 4, 5], index=[5.43, 6.34, 7.34, 8.0, 9.1])
    gs = cudf.Series.from_pandas(ps)

    assert_eq(ps.loc[5.43], gs.loc[5.43])
    assert_eq(ps.loc[8], gs.loc[8])
    assert_eq(ps.loc[6.1:8], gs.loc[6.1:8])
    assert_eq(ps.loc[:7.1], gs.loc[:7.1])
    assert_eq(ps.loc[6.345:], gs.loc[6.345:])
    assert_eq(ps.loc[::2], gs.loc[::2])
    assert_eq(
        ps.loc[[True, False, True, False, True]],
        gs.loc[[True, False, True, False, True]],
    )


def test_series_loc_string():
    ps = pd.Series(
        [1, 2, 3, 4, 5], index=["one", "two", "three", "four", "five"]
    )
    gs = cudf.Series.from_pandas(ps)

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
    gs = cudf.Series.from_pandas(ps)

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
        ps.loc["2001-01-02":"2001-01-05"],
        gs.loc["2001-01-02":"2001-01-05"],
        check_freq=False,
    )
    assert_eq(ps.loc["2001-01-02":], gs.loc["2001-01-02":], check_freq=False)
    assert_eq(ps.loc[:"2001-01-04"], gs.loc[:"2001-01-04"], check_freq=False)
    assert_eq(ps.loc[::2], gs.loc[::2], check_freq=False)

    assert_eq(
        ps.loc[["2001-01-01", "2001-01-04", "2001-01-05"]],
        gs.loc[["2001-01-01", "2001-01-04", "2001-01-05"]],
    )

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
        check_freq=False,
    )

    just_less_than_max = ps.index.max() - pd.Timedelta("5m")

    assert_eq(
        ps.loc[:just_less_than_max],
        gs.loc[:just_less_than_max],
        check_freq=False,
    )


def test_series_loc_categorical():
    ps = pd.Series(
        [1, 2, 3, 4, 5], index=pd.Categorical(["a", "b", "c", "d", "e"])
    )
    gs = cudf.Series.from_pandas(ps)

    assert_eq(ps.loc["a"], gs.loc["a"])
    assert_eq(ps.loc["e"], gs.loc["e"])
    assert_eq(ps.loc["b":"d"], gs.loc["b":"d"])
    assert_eq(ps.loc[:"d"], gs.loc[:"d"])
    assert_eq(ps.loc["b":], gs.loc["b":])
    assert_eq(ps.loc[::2], gs.loc[::2])

    # order of categories changes, so we can only
    # compare values:
    assert_eq(
        ps.loc[["a", "d", "e"]].values, gs.loc[["a", "d", "e"]].to_numpy()
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
    # create random cudf.Series
    np.random.seed(12)
    ps = pd.Series(np.random.sample(nelem))

    # gpu cudf.Series
    gs = cudf.Series(ps)

    # positive tests for indexing
    np.testing.assert_allclose(gs.iloc[-1 * nelem], ps.iloc[-1 * nelem])
    np.testing.assert_allclose(gs.iloc[-1], ps.iloc[-1])
    np.testing.assert_allclose(gs.iloc[0], ps.iloc[0])
    np.testing.assert_allclose(gs.iloc[1], ps.iloc[1])
    np.testing.assert_allclose(gs.iloc[nelem - 1], ps.iloc[nelem - 1])

    # positive tests for slice
    np.testing.assert_allclose(gs.iloc[-1:1].to_numpy(), ps.iloc[-1:1])
    np.testing.assert_allclose(
        gs.iloc[nelem - 1 : -1].to_numpy(), ps.iloc[nelem - 1 : -1]
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
    gdf = cudf.DataFrame()

    gdf["a"] = ha = np.random.randint(low=0, high=100, size=nelem).astype(
        np.int32
    )
    gdf["b"] = hb = np.random.random(nelem).astype(np.float32)

    pdf = pd.DataFrame()
    pdf["a"] = ha
    pdf["b"] = hb

    gdf.index.name = "index"
    pdf.index.name = "index"

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


def test_dataframe_iloc_tuple():
    gdf = cudf.DataFrame()
    nelem = 123
    gdf["a"] = ha = np.random.randint(low=0, high=100, size=nelem).astype(
        np.int32
    )
    gdf["b"] = hb = np.random.random(nelem).astype(np.float32)

    pdf = pd.DataFrame()
    pdf["a"] = ha
    pdf["b"] = hb

    assert_eq(gdf.iloc[1, [1]], pdf.iloc[1, [1]], check_dtype=False)
    assert_eq(gdf.iloc[:, -1], pdf.iloc[:, -1])


def test_dataframe_iloc_index_error():
    gdf = cudf.DataFrame()
    nelem = 123
    gdf["a"] = ha = np.random.randint(low=0, high=100, size=nelem).astype(
        np.int32
    )
    gdf["b"] = hb = np.random.random(nelem).astype(np.float32)

    pdf = pd.DataFrame()
    pdf["a"] = ha
    pdf["b"] = hb

    with pytest.raises(IndexError):
        pdf.iloc[nelem * 2]
    with pytest.raises(IndexError):
        gdf.iloc[nelem * 2]


@pytest.mark.parametrize("ntake", [0, 1, 10, 123, 122, 200])
def test_dataframe_take(ntake):
    np.random.seed(0)
    df = cudf.DataFrame()

    nelem = 123
    df["ii"] = np.random.randint(0, 20, nelem)
    df["ff"] = np.random.random(nelem)

    take_indices = np.random.randint(0, len(df), ntake)

    actual = df.take(take_indices)
    expected = df.to_pandas().take(take_indices)

    assert actual.ii.null_count == 0
    assert actual.ff.null_count == 0
    assert_eq(actual, expected)


@pytest.mark.parametrize("ntake", [1, 2, 8, 9])
def test_dataframe_take_with_multiindex(ntake):
    np.random.seed(0)
    df = cudf.DataFrame(
        index=cudf.MultiIndex(
            levels=[["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
    )

    nelem = 9
    df["ii"] = np.random.randint(0, 20, nelem)
    df["ff"] = np.random.random(nelem)

    take_indices = np.random.randint(0, len(df), ntake)

    actual = df.take(take_indices)
    expected = df.to_pandas().take(take_indices)

    assert_eq(actual, expected)


@pytest.mark.parametrize("ntake", [0, 1, 10, 123, 122, 200])
def test_series_take(ntake):
    np.random.seed(0)
    nelem = 123

    psr = pd.Series(np.random.randint(0, 20, nelem))
    gsr = cudf.Series(psr)

    take_indices = np.random.randint(0, len(gsr), ntake)

    actual = gsr.take(take_indices)
    expected = psr.take(take_indices)

    assert_eq(actual, expected)


def test_series_take_positional():
    psr = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])

    gsr = cudf.Series.from_pandas(psr)

    take_indices = [1, 2, 0, 3]

    expect = psr.take(take_indices)
    got = gsr.take(take_indices)

    assert_eq(expect, got)


@pytest.mark.parametrize("nelem", [0, 1, 5, 20, 100])
@pytest.mark.parametrize("slice_start", [None, 0, 1, 3, 10, -10])
@pytest.mark.parametrize("slice_end", [None, 0, 1, 30, 50, -1])
def test_dataframe_masked_slicing(nelem, slice_start, slice_end):
    gdf = cudf.DataFrame()
    gdf["a"] = list(range(nelem))
    gdf["b"] = list(range(nelem, 2 * nelem))
    gdf["a"] = gdf["a"]._column.set_mask(utils.random_bitmask(nelem))
    gdf["b"] = gdf["b"]._column.set_mask(utils.random_bitmask(nelem))

    def do_slice(x):
        return x[slice_start:slice_end]

    expect = do_slice(gdf.to_pandas())
    got = do_slice(gdf).to_pandas()

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("dtype", [int, float, str])
def test_empty_boolean_mask(dtype):
    gdf = cudf.datasets.randomdata(nrows=0, dtypes={"a": dtype})
    pdf = gdf.to_pandas()

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
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert_eq(pdf[[True, False, True, False]], gdf[[True, False, True, False]])


"""
This test compares cudf and Pandas DataFrame boolean indexing.
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


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
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
    with expect_warning_if(
        isinstance(value, list) and len(value) == 0 and nulls == "none"
    ):
        psr[key] = value
    with expect_warning_if(
        isinstance(value, list) and len(value) == 0 and not len(key) == 0
    ):
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


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
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
    with expect_warning_if(
        isinstance(value, list) and len(value) == 0 and nulls == "none"
    ):
        psr.iloc[key] = value
    with expect_warning_if(
        isinstance(value, list) and len(value) == 0 and not len(key) == 0
    ):
        gsr.iloc[key] = value
    assert_eq(psr, gsr, check_dtype=False)


@pytest.mark.parametrize(
    "key, value",
    [
        pytest.param(
            0,
            0.5,
        ),
        ([0, 1], 0.5),
        ([0, 1], [0.5, 2.5]),
        (slice(0, 2), [0.5, 0.25]),
    ],
)
def test_series_setitem_dtype(key, value):
    psr = pd.Series([1, 2, 3], dtype="int32")
    gsr = cudf.from_pandas(psr)

    with expect_warning_if(isinstance(value, (float, list))):
        psr[key] = value
    with expect_warning_if(isinstance(value, (float, list))):
        gsr[key] = value

    assert_eq(psr, gsr)


def test_series_setitem_datetime():
    psr = pd.Series(["2001", "2002", "2003"], dtype="datetime64[ns]")
    gsr = cudf.from_pandas(psr)

    psr[0] = np.datetime64("2005")
    gsr[0] = np.datetime64("2005")

    assert_eq(psr, gsr)


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
        ("b", np.int8(8)),
        ("d", 4),
        ("d", np.int8(16)),
        ("d", np.float32(16)),
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
        (1, "d"),
        (2, "e"),
        (4, "f"),
        ([1, 3], "g"),
        ([1, 3], ["g", "h"]),
        ([True, False, True], "i"),
        ([False, False, False], "j"),
        ([True, False, True], ["k", "l"]),
    ],
)
def test_series_setitem_loc_numeric_index(key, value):
    psr = pd.Series(["a", "b", "c"], [1, 2, 3])
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
        ((slice(None), "a"), [3, 2, 1]),
        ((slice(None, "two"), "a"), range(2)),
        ((slice(None, "two"), "a"), [4, 5]),
        ((["one", "two"], "a"), 5),
        (("one", "c"), 5),
        ((["one", "two"], "c"), 5),
        ((slice(None), "c"), 5),
        ((slice(None), "c"), range(3)),
        ((slice(None), "c"), [3, 2, 1]),
        ((slice(None, "two"), "c"), range(2)),
        ((slice(None, "two"), "c"), [4, 5]),
    ],
)
def test_dataframe_setitem_loc(key, value, pdf_gdf):
    pdf, gdf = pdf_gdf
    pdf.loc[key] = value
    gdf.loc[key] = value
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "key, value",
    [
        (("one", "a"), 5),
        ((slice(None), "a"), range(3)),
        ((slice(None), "a"), [3, 2, 1]),
    ],
)
def test_dataframe_setitem_loc_empty_df(key, value):
    pdf, gdf = pd.DataFrame(), cudf.DataFrame()
    pdf.loc[key] = value
    gdf.loc[key] = value
    assert_eq(pdf, gdf, check_dtype=False)


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
    psr = pd.Series([1, 2, 3])
    gsr = cudf.from_pandas(psr)

    assert_exceptions_equal(
        lambda: psr[[0, 1, 9]],
        lambda: gsr[[0, 1, 9]],
    )
    assert_exceptions_equal(
        lambda: psr[[0, 1, -4]],
        lambda: gsr[[0, 1, -4]],
    )
    assert_exceptions_equal(
        lambda: psr.__setitem__([0, 1, 9], 2),
        lambda: gsr.__setitem__([0, 1, 9], 2),
    )
    assert_exceptions_equal(
        lambda: psr.__setitem__([0, 1, -4], 2),
        lambda: gsr.__setitem__([0, 1, -4], 2),
    )


def test_out_of_bounds_indexing_empty():
    psr = pd.Series(dtype="int64")
    gsr = cudf.from_pandas(psr)
    assert_exceptions_equal(
        lambda: psr.iloc.__setitem__(-1, 2),
        lambda: gsr.iloc.__setitem__(-1, 2),
    )
    assert_exceptions_equal(
        lambda: psr.iloc.__setitem__(1, 2),
        lambda: gsr.iloc.__setitem__(1, 2),
    )


def test_sliced_indexing():
    a = list(range(4, 4 + 150))
    b = list(range(0, 0 + 150))
    pdf = pd.DataFrame({"a": a, "b": b})
    gdf = cudf.DataFrame.from_pandas(pdf)
    pdf = pdf.set_index("a")
    gdf = gdf.set_index("a")
    pidx = pdf.index[:75]
    gidx = gdf.index[:75]

    assert_eq(pdf.loc[pidx], gdf.loc[gidx])


@pytest.mark.parametrize("index", [["a"], ["a", "a"], ["a", "a", "b", "c"]])
def test_iloc_categorical_index(index):
    gdf = cudf.DataFrame({"data": range(len(index))}, index=index)
    gdf.index = gdf.index.astype("category")
    pdf = gdf.to_pandas()
    expect = pdf.iloc[:, 0]
    got = gdf.iloc[:, 0]
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "sli",
    [
        slice("2001", "2002"),
        slice("2002", "2001"),
        slice("2001", None),
    ],
)
@pytest.mark.parametrize("is_dataframe", [True, False])
def test_loc_datetime_index(sli, is_dataframe):
    sli = slice(pd.to_datetime(sli.start), pd.to_datetime(sli.stop))

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
    "sli",
    [
        slice("2001", "2020"),
        slice(None, "2020"),
    ],
)
def test_loc_datetime_index_slice_not_in(sli):
    pd_data = pd.Series(
        [1, 2, 3],
        pd.Series(["2001", "2009", "2002"], dtype="datetime64[ns]"),
    )
    gd_data = cudf.from_pandas(pd_data)
    with pytest.raises(KeyError):
        assert_eq(pd_data.loc[sli], gd_data.loc[sli])

    with pytest.raises(KeyError):
        sli = slice(pd.to_datetime(sli.start), pd.to_datetime(sli.stop))
        assert_eq(pd_data.loc[sli], gd_data.loc[sli])


@pytest.mark.parametrize(
    "gdf_kwargs",
    [
        {"data": {"a": range(1000)}},
        {"data": {"a": range(1000), "b": range(1000)}},
        {
            "data": {
                "a": range(20),
                "b": range(20),
                "c": ["abc", "def", "xyz", "def", "pqr"] * 4,
            }
        },
        {"index": [1, 2, 3]},
        {"index": range(1000)},
        {"columns": ["a", "b", "c", "d"]},
        {"columns": ["a"], "index": range(1000)},
        {"columns": ["a", "col2", "...col n"], "index": range(1000)},
        {"index": cudf.Series(range(1000)).astype("str")},
        {
            "columns": ["a", "b", "c", "d"],
            "index": cudf.Series(range(1000)).astype("str"),
        },
    ],
)
@pytest.mark.parametrize(
    "slice",
    [
        slice(6, None),  # start but no stop, [6:]
        slice(None, None, 3),  # only step, [::3]
        slice(1, 10, 2),  # start, stop, step
        slice(3, -5, 2),  # negative stop
        slice(-2, -4),  # slice is empty
        slice(-10, -20, -1),  # reversed slice
        slice(None),  # slices everything, same as [:]
        slice(250, 500),
        slice(250, 251),
        slice(50),
        slice(1, 10),
        slice(10, 20),
        slice(15, 24),
        slice(6),
    ],
)
def test_dataframe_sliced(gdf_kwargs, slice):
    gdf = cudf.DataFrame(**gdf_kwargs)
    pdf = gdf.to_pandas()

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
    "slice",
    [slice(6), slice(1), slice(7), slice(1, 3)],
)
def test_dataframe_iloc_index(gdf, slice):
    pdf = gdf.to_pandas()

    actual = gdf.iloc[:, slice]
    expected = pdf.iloc[:, slice]

    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        [[0], [1], [2]],
        [[0, 1], [2, 3], [4, 5]],
        [[[0, 1], [2]], [[3, 4]], [[5, 6]]],
        [None, [[0, 1], [2]], [[3, 4], [5, 6]]],
        [[], [[0, 1], [2]], [[3, 4], [5, 6]]],
        [[], [["a", "b"], None], [["c", "d"], []]],
    ],
)
@pytest.mark.parametrize(
    "key", [[], [0], [0, 1], [0, 1, 0], slice(None), slice(0, 2), slice(1, 3)]
)
def test_iloc_with_lists(data, key):
    psr = pd.Series(data)
    gsr = cudf.Series(data)
    assert_eq(psr.iloc[key], gsr.iloc[key])

    pdf = pd.DataFrame({"a": data, "b": data})
    gdf = cudf.DataFrame({"a": data, "b": data})
    assert_eq(pdf.iloc[key], gdf.iloc[key])


@pytest.mark.parametrize("key", [5, -10, "0", "a", np.array(5), np.array("a")])
def test_loc_bad_key_type(key):
    psr = pd.Series([1, 2, 3])
    gsr = cudf.from_pandas(psr)
    assert_exceptions_equal(lambda: psr[key], lambda: gsr[key])
    assert_exceptions_equal(lambda: psr.loc[key], lambda: gsr.loc[key])


@pytest.mark.parametrize("key", ["b", 1.0, np.array("b")])
def test_loc_bad_key_type_string_index(key):
    psr = pd.Series([1, 2, 3], index=["a", "1", "c"])
    gsr = cudf.from_pandas(psr)
    assert_exceptions_equal(lambda: psr[key], lambda: gsr[key])
    assert_exceptions_equal(lambda: psr.loc[key], lambda: gsr.loc[key])


def test_loc_zero_dim_array():
    psr = pd.Series([1, 2, 3])
    gsr = cudf.from_pandas(psr)

    assert_eq(psr[np.array(0)], gsr[np.array(0)])
    assert_eq(psr[np.array([0])[0]], gsr[np.array([0])[0]])


@pytest.mark.parametrize(
    "arg",
    [
        slice(None),
        slice((1, 2), None),
        slice(None, (1, 2)),
        (1, 1),
        pytest.param(
            (1, slice(None)),
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/46704"
            ),
        ),
        1,
        2,
    ],
)
def test_loc_series_multiindex(arg):
    gsr = cudf.DataFrame(
        {"a": [1, 1, 2], "b": [1, 2, 3], "c": ["a", "b", "c"]}
    ).set_index(["a", "b"])["c"]
    psr = gsr.to_pandas()
    assert_eq(psr.loc[arg], gsr.loc[arg])


@pytest.mark.parametrize(
    "arg",
    [
        slice(None, None, -1),
        slice(None, -1, -1),
        slice(4, -1, -1),
        slice(None, None, -3),
        slice(None, -1, -3),
        slice(4, -1, -3),
    ],
)
@pytest.mark.parametrize(
    "pobj", [pd.DataFrame({"a": [1, 2, 3, 4, 5]}), pd.Series([1, 2, 3, 4, 5])]
)
def test_iloc_before_zero_terminate(arg, pobj):
    gobj = cudf.from_pandas(pobj)

    assert_eq(pobj.iloc[arg], gobj.iloc[arg])


def test_iloc_decimal():
    sr = cudf.Series(["1.00", "2.00", "3.00", "4.00"]).astype(
        cudf.Decimal64Dtype(scale=2, precision=3)
    )
    got = sr.iloc[[3, 2, 1, 0]]
    expect = cudf.Series(
        ["4.00", "3.00", "2.00", "1.00"],
    ).astype(cudf.Decimal64Dtype(scale=2, precision=3))
    assert_eq(expect.reset_index(drop=True), got.reset_index(drop=True))


@pytest.mark.parametrize(
    ("key, value"),
    [
        (
            ([0], ["x", "y"]),
            [10, 20],
        ),
        (
            ([0, 2], ["x", "y"]),
            [[10, 30], [20, 40]],
        ),
        (
            (0, ["x", "y"]),
            [10, 20],
        ),
        (
            ([0, 2], "x"),
            [10, 20],
        ),
    ],
)
def test_dataframe_loc_inplace_update(key, value):
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    pdf = gdf.to_pandas()

    actual = gdf.loc[key] = value
    expected = pdf.loc[key] = value

    assert_eq(expected, actual)


def test_dataframe_loc_inplace_update_string_index():
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=list("abc"))
    pdf = gdf.to_pandas()

    actual = gdf.loc[["a"], ["x", "y"]] = [10, 20]
    expected = pdf.loc[["a"], ["x", "y"]] = [10, 20]

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    ("key, value"),
    [
        ([0], [10, 20]),
        ([0, 2], [[10, 30], [20, 40]]),
        (([0, 2], [0, 1]), [[10, 30], [20, 40]]),
        (([0, 2], 0), [10, 30]),
        ((0, [0, 1]), [20, 40]),
    ],
)
def test_dataframe_iloc_inplace_update(key, value):
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    pdf = gdf.to_pandas()

    actual = gdf.iloc[key] = value
    expected = pdf.iloc[key] = value

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "loc_key",
    [([0, 2], ["x", "y"])],
)
@pytest.mark.parametrize(
    "iloc_key",
    [[0, 2]],
)
@pytest.mark.parametrize(
    ("data, index"),
    [
        (
            {"x": [10, 20], "y": [30, 40]},
            [0, 2],
        )
    ],
)
def test_dataframe_loc_iloc_inplace_update_with_RHS_dataframe(
    loc_key, iloc_key, data, index
):
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    pdf = gdf.to_pandas()

    actual = gdf.loc[loc_key] = cudf.DataFrame(data, index=cudf.Index(index))
    expected = pdf.loc[loc_key] = pd.DataFrame(data, index=pd.Index(index))
    assert_eq(expected, actual)

    actual = gdf.iloc[iloc_key] = cudf.DataFrame(data, index=cudf.Index(index))
    expected = pdf.iloc[iloc_key] = pd.DataFrame(data, index=pd.Index(index))
    assert_eq(expected, actual)


def test_dataframe_loc_inplace_update_with_invalid_RHS_df_columns():
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    pdf = gdf.to_pandas()

    actual = gdf.loc[[0, 2], ["x", "y"]] = cudf.DataFrame(
        {"b": [10, 20], "y": [30, 40]}, index=cudf.Index([0, 2])
    )
    with pytest.warns(FutureWarning):
        # Seems to be a false warning from pandas,
        # but nevertheless catching it.
        expected = pdf.loc[[0, 2], ["x", "y"]] = pd.DataFrame(
            {"b": [10, 20], "y": [30, 40]}, index=pd.Index([0, 2])
        )

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    ("key, value"),
    [
        (([0, 2], ["x", "y"]), [[10, 30, 50], [20, 40, 60]]),
        (([0], ["x", "y"]), [[10], [20]]),
    ],
)
def test_dataframe_loc_inplace_update_shape_mismatch(key, value):
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(ValueError, match="shape mismatch:"):
        gdf.loc[key] = value


@pytest.mark.parametrize(
    ("key, value"),
    [
        ([0, 2], [[10, 30, 50], [20, 40, 60]]),
        ([0], [[10], [20]]),
    ],
)
def test_dataframe_iloc_inplace_update_shape_mismatch(key, value):
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(ValueError, match="shape mismatch:"):
        gdf.iloc[key] = value


def test_dataframe_loc_inplace_update_shape_mismatch_RHS_df():
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(ValueError, match="shape mismatch:"):
        gdf.loc[([0, 2], ["x", "y"])] = cudf.DataFrame(
            {"x": [10, 20]}, index=cudf.Index([0, 2])
        )


def test_dataframe_iloc_inplace_update_shape_mismatch_RHS_df():
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(ValueError, match="shape mismatch:"):
        gdf.iloc[[0, 2]] = cudf.DataFrame(
            {"x": [10, 20]}, index=cudf.Index([0, 2])
        )


@pytest.mark.parametrize(
    "array,is_error",
    [
        (cupy.arange(20, 40).reshape(-1, 2), False),
        (cupy.arange(20, 50).reshape(-1, 3), True),
        (np.arange(20, 40).reshape(-1, 2), False),
        (np.arange(20, 30).reshape(-1, 1), False),
        (cupy.arange(20, 30).reshape(-1, 1), False),
    ],
)
def test_dataframe_indexing_setitem_np_cp_array(array, is_error):
    gdf = cudf.DataFrame({"a": range(10), "b": range(10)})
    pdf = gdf.to_pandas()
    if not is_error:
        gdf.loc[:, ["a", "b"]] = array
        pdf.loc[:, ["a", "b"]] = cupy.asnumpy(array)

        assert_eq(gdf, pdf)
    else:
        assert_exceptions_equal(
            lfunc=pdf.loc.__setitem__,
            rfunc=gdf.loc.__setitem__,
            lfunc_args_and_kwargs=(
                [(slice(None, None, None), ["a", "b"]), cupy.asnumpy(array)],
                {},
            ),
            rfunc_args_and_kwargs=(
                [(slice(None, None, None), ["a", "b"]), array],
                {},
            ),
        )


def test_iloc_single_row_with_nullable_column():
    # see https://github.com/rapidsai/cudf/issues/11349
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.4]})
    df = cudf.from_pandas(pdf)

    df.iloc[0]  # before the fix for #11349 this would segfault
    assert_eq(pdf.iloc[0], df.iloc[0])


def test_loc_single_row_from_slice():
    # see https://github.com/rapidsai/cudf/issues/11930
    pdf = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]}).set_index("a")
    df = cudf.from_pandas(pdf)
    assert_eq(pdf.loc[5:10], df.loc[5:10])


@pytest.mark.parametrize("indexer", ["loc", "iloc"])
@pytest.mark.parametrize(
    "mask",
    [[False, True], [False, False, True, True, True]],
    ids=["too-short", "too-long"],
)
def test_boolean_mask_wrong_length(indexer, mask):
    s = pd.Series([1, 2, 3, 4])

    indexee = getattr(s, indexer)
    with pytest.raises(IndexError):
        indexee[mask]

    c = cudf.from_pandas(s)
    indexee = getattr(c, indexer)
    with pytest.raises(IndexError):
        indexee[mask]


@pytest.mark.parametrize("indexer", ["loc", "iloc"])
def test_boolean_mask_columns(indexer):
    df = pd.DataFrame(np.zeros((3, 3)))
    cdf = cudf.from_pandas(df)
    mask = [True, False, True]
    expect = getattr(df, indexer)[:, mask]
    got = getattr(cdf, indexer)[:, mask]

    assert_eq(expect, got)


@pytest.mark.parametrize("indexer", ["loc", "iloc"])
@pytest.mark.parametrize(
    "mask",
    [[False, True], [False, False, True, True, True]],
    ids=["too-short", "too-long"],
)
def test_boolean_mask_columns_wrong_length(indexer, mask):
    df = pd.DataFrame(np.zeros((3, 3)))
    cdf = cudf.from_pandas(df)

    with pytest.raises(IndexError):
        getattr(df, indexer)[:, mask]
    with pytest.raises(IndexError):
        getattr(cdf, indexer)[:, mask]


def test_boolean_mask_columns_iloc_series():
    df = pd.DataFrame(np.zeros((3, 3)))
    cdf = cudf.from_pandas(df)

    mask = pd.Series([True, False, True], dtype=bool)
    with pytest.raises(NotImplementedError):
        df.iloc[:, mask]

    with pytest.raises(NotImplementedError):
        cdf.iloc[:, mask]


@pytest.mark.parametrize("index_type", ["single", "slice"])
def test_loc_timestamp_issue_8585(index_type):
    # https://github.com/rapidsai/cudf/issues/8585
    start = pd.Timestamp(
        datetime.strptime("2021-03-12 00:00", "%Y-%m-%d %H:%M")
    )
    end = pd.Timestamp(datetime.strptime("2021-03-12 11:00", "%Y-%m-%d %H:%M"))
    timestamps = pd.date_range(start, end, periods=12)
    value = np.random.normal(size=12)
    df = pd.DataFrame(value, index=timestamps, columns=["value"])
    cdf = cudf.from_pandas(df)
    if index_type == "single":
        index = pd.Timestamp(
            datetime.strptime("2021-03-12 03:00", "%Y-%m-%d %H:%M")
        )
    elif index_type == "slice":
        index = slice(start, end, None)
    else:
        raise ValueError("Invalid index type")
    expect = df.loc[index]
    actual = cdf.loc[index]
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "index_type",
    [
        "single",
        pytest.param(
            "slice",
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/8585"
            ),
        ),
        pytest.param(
            "date_range",
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/8585"
            ),
        ),
    ],
)
def test_loc_multiindex_timestamp_issue_8585(index_type):
    # https://github.com/rapidsai/cudf/issues/8585
    start = pd.Timestamp(
        datetime.strptime("2021-03-12 00:00", "%Y-%m-%d %H:%M")
    )
    end = pd.Timestamp(datetime.strptime("2021-03-12 03:00", "%Y-%m-%d %H:%M"))
    timestamps = pd.date_range(start, end, periods=4)
    labels = ["A", "B", "C"]
    index = pd.MultiIndex.from_product(
        [timestamps, labels], names=["timestamp", "label"]
    )
    value = np.random.normal(size=12)
    df = pd.DataFrame(value, index=index, columns=["value"])
    cdf = cudf.from_pandas(df)
    start = pd.Timestamp(
        datetime.strptime("2021-03-12 01:00", "%Y-%m-%d %H:%M")
    )
    end = pd.Timestamp(datetime.strptime("2021-03-12 02:00", "%Y-%m-%d %H:%M"))
    if index_type == "single":
        index = pd.Timestamp(
            datetime.strptime("2021-03-12 03:00", "%Y-%m-%d %H:%M")
        )
    elif index_type == "slice":
        index = slice(start, end, None)
    elif index_type == "date_range":
        index = pd.date_range(start, end, periods=2)
    else:
        raise ValueError("Invalid index type")
    expect = df.loc[index]
    actual = cdf.loc[index]
    assert_eq(expect, actual)


def test_loc_repeated_index_label_issue_8693():
    # https://github.com/rapidsai/cudf/issues/8693
    s = pd.Series([1, 2, 3, 4], index=[0, 1, 1, 2])
    cs = cudf.from_pandas(s)
    expect = s.loc[1]
    actual = cs.loc[1]
    assert_eq(expect, actual)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/13268")
@pytest.mark.parametrize(
    "indexer", [(..., 0), (0, ...)], ids=["row_ellipsis", "column_ellipsis"]
)
def test_loc_ellipsis_as_slice_issue_13268(indexer):
    # https://github.com/rapidsai/cudf/issues/13268
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)

    expect = df.loc[indexer]
    actual = cdf.loc[indexer]
    assert_eq(expect, actual)


@pytest.mark.xfail(
    reason="https://github.com/rapidsai/cudf/issues/13269 "
    "and https://github.com/rapidsai/cudf/issues/13273"
)
def test_loc_repeated_column_label_issue_13269():
    # https://github.com/rapidsai/cudf/issues/13269
    # https://github.com/rapidsai/cudf/issues/13273
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)

    expect = df.loc[:, [0, 1, 0]]
    actual = cdf.loc[:, [0, 1, 0]]
    assert_eq(expect, actual)


def test_loc_column_boolean_mask_issue_13270():
    # https://github.com/rapidsai/cudf/issues/13270
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)
    expect = df.loc[:, [True, True]]
    actual = cdf.loc[:, [True, True]]
    assert_eq(expect, actual)


@pytest.mark.parametrize("indexer", [[1], [0, 2]])
def test_iloc_integer_categorical_issue_13013(indexer):
    # https://github.com/rapidsai/cudf/issues/13013
    s = pd.Series([0, 1, 2])
    index = pd.Categorical(indexer)
    expect = s.iloc[index]
    c = cudf.from_pandas(s)
    actual = c.iloc[index]
    assert_eq(expect, actual)


def test_iloc_incorrect_boolean_mask_length_issue_13015():
    # https://github.com/rapidsai/cudf/issues/13015
    s = pd.Series([0, 1, 2])
    with pytest.raises(IndexError):
        s.iloc[[True, False]]
    c = cudf.from_pandas(s)
    with pytest.raises(IndexError):
        c.iloc[[True, False]]


def test_iloc_column_boolean_mask_issue_13265():
    # https://github.com/rapidsai/cudf/issues/13265
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)
    expect = df.iloc[:, [True, True]]
    actual = cdf.iloc[:, [True, True]]
    assert_eq(expect, actual)


def test_iloc_repeated_column_label_issue_13266():
    # https://github.com/rapidsai/cudf/issues/13266
    # https://github.com/rapidsai/cudf/issues/13273
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)

    with pytest.raises(NotImplementedError):
        cdf.iloc[:, [0, 1, 0]]


@pytest.mark.parametrize(
    "indexer",
    [
        (..., 0),
        (0, ...),
    ],
    ids=["row_ellipsis", "column_ellipsis"],
)
def test_iloc_ellipsis_as_slice_issue_13267(indexer):
    # https://github.com/rapidsai/cudf/issues/13267
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)

    expect = df.iloc[indexer]
    actual = cdf.iloc[indexer]
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "indexer",
    [
        0,
        (slice(None), 0),
        ([0, 2], 1),
        (slice(None), slice(None)),
        (slice(None), [1, 0]),
        (0, 0),
        (1, [1, 0]),
        ([1, 0], 0),
        ([1, 2], [0, 1]),
    ],
)
def test_iloc_multiindex_lookup_as_label_issue_13515(indexer):
    # https://github.com/rapidsai/cudf/issues/13515
    df = pd.DataFrame(
        {"a": [1, 1, 3], "b": [2, 3, 4], "c": [1, 6, 7], "d": [1, 8, 9]}
    ).set_index(["a", "b"])
    cdf = cudf.from_pandas(df)

    expect = df.iloc[indexer]
    actual = cdf.iloc[indexer]
    assert_eq(expect, actual)


def test_loc_unsorted_index_slice_lookup_keyerror_issue_12833():
    # https://github.com/rapidsai/cudf/issues/12833
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[7, 0, 4])
    cdf = cudf.from_pandas(df)

    # Check that pandas don't change their mind
    with pytest.raises(KeyError):
        df.loc[1:5]

    with pytest.raises(KeyError):
        cdf.loc[1:5]


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/13379")
@pytest.mark.parametrize("index", [range(5), list(range(5))])
def test_loc_missing_label_keyerror_issue_13379(index):
    # https://github.com/rapidsai/cudf/issues/13379
    df = pd.DataFrame({"a": index}, index=index)
    cdf = cudf.from_pandas(df)
    # Check that pandas don't change their mind
    with pytest.raises(KeyError):
        df.loc[[0, 5]]

    with pytest.raises(KeyError):
        cdf.loc[[0, 5]]


@pytest.mark.parametrize("series", [True, False], ids=["Series", "DataFrame"])
def test_loc_repeated_label_ordering_issue_13658(series):
    # https://github.com/rapidsai/cudf/issues/13658
    values = range(2048)
    index = [1 for _ in values]
    if series:
        frame = cudf.Series(values, index=index)
    else:
        frame = cudf.DataFrame({"a": values}, index=index)
    expect = frame.to_pandas().loc[[1]]
    actual = frame.loc[[1]]
    assert_eq(actual, expect)


@pytest.mark.parametrize("index", [None, [2, 1, 3, 5, 4]])
def test_loc_bool_key_numeric_index_raises(index):
    ser = cudf.Series(range(5), index=index)
    with pytest.raises(KeyError):
        ser.loc[True]


class TestLocIndexWithOrder:
    # https://github.com/rapidsai/cudf/issues/12833
    @pytest.fixture(params=["increasing", "decreasing", "neither"])
    def order(self, request):
        return request.param

    @pytest.fixture(params=[-1, 1], ids=["reverse", "forward"])
    def take_order(self, request):
        return request.param

    @pytest.fixture(params=["float", "int", "string", "range"])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def index(self, order, dtype):
        if dtype == "string":
            index = ["a", "h", "f", "z"]
        elif dtype == "int":
            index = [-1, 10, 7, 14]
        elif dtype == "float":
            index = [-1.5, 7.10, 2.4, 11.2]
        elif dtype == "range":
            if order == "increasing":
                return cudf.RangeIndex(2, 10, 3)
            elif order == "decreasing":
                return cudf.RangeIndex(10, 1, -3)
            else:
                return cudf.RangeIndex(10, 20, 3)
        else:
            raise ValueError(f"Unhandled index dtype {dtype}")
        if order == "decreasing":
            return sorted(index, reverse=True)
        elif order == "increasing":
            return sorted(index)
        elif order == "neither":
            return index
        else:
            raise ValueError(f"Unhandled index order {order}")

    @pytest.fixture
    def df(self, index):
        return cudf.DataFrame({"a": range(len(index))}, index=index)

    def test_loc_index_inindex_slice(self, df, take_order):
        pdf = df.to_pandas()
        lo = pdf.index[1]
        hi = pdf.index[-2]
        expect = pdf.loc[lo:hi:take_order]
        actual = df.loc[lo:hi:take_order]
        assert_eq(expect, actual)

    def test_loc_index_inindex_subset(self, df, take_order):
        pdf = df.to_pandas()
        vals = [pdf.index[0], pdf.index[2]][::take_order]
        expect = pdf.loc[vals]
        actual = df.loc[vals]
        assert_eq(expect, actual)

    def test_loc_index_notinindex_slice(
        self, request, df, order, dtype, take_order
    ):
        pdf = df.to_pandas()
        lo = pdf.index[1]
        hi = pdf.index[-2]
        if isinstance(lo, str):
            lo = chr(ord(lo) - 1)
            hi = chr(ord(hi) + 1)
        else:
            lo -= 1
            hi += 1
        if order == "neither" and dtype != "range":
            with pytest.raises(KeyError):
                pdf.loc[lo:hi:take_order]
            with pytest.raises(KeyError):
                df.loc[lo:hi:take_order]
        else:
            expect = pdf.loc[lo:hi:take_order]
            actual = df.loc[lo:hi:take_order]
            assert_eq(expect, actual)


@pytest.mark.parametrize(
    "arg",
    [
        (2, ("one", "second")),
        (slice(None, None, None), ("two", "first")),
        (1, ("one", "first")),
        (slice(None, None, None), ("two", "second")),
        (slice(None, None, None), ("two", "first", "three")),
        (3, ("two", "first", "three")),
        (slice(None, None, None), ("two",)),
        (0, ("two",)),
    ],
)
def test_loc_dataframe_column_multiindex(arg):
    gdf = cudf.DataFrame(
        [list("abcd"), list("efgh"), list("ijkl"), list("mnop")],
        columns=cudf.MultiIndex.from_product(
            [["one", "two"], ["first", "second"], ["three"]]
        ),
    )
    pdf = gdf.to_pandas()

    assert_eq(gdf.loc[arg], pdf.loc[arg])


@pytest.mark.parametrize(
    "arg", [slice(2, 4), slice(2, 5), slice(2.3, 5), slice(4.6, 6)]
)
def test_series_iloc_float_int(arg):
    gs = cudf.Series(range(4), index=[2.0, 3.0, 4.5, 5.5])
    ps = gs.to_pandas()

    actual = gs.loc[arg]
    expected = ps.loc[arg]

    assert_eq(actual, expected)


def test_iloc_loc_mixed_dtype():
    df = cudf.DataFrame({"a": ["a", "b"], "b": [0, 1]})
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(TypeError):
            df.iloc[0]
        with pytest.raises(TypeError):
            df.loc[0]
    df = df.astype("str")
    pdf = df.to_pandas()

    assert_eq(df.iloc[0], pdf.iloc[0])
    assert_eq(df.loc[0], pdf.loc[0])


def test_loc_setitem_categorical_integer_not_position_based():
    gdf = cudf.DataFrame(range(3), index=cudf.CategoricalIndex([1, 2, 3]))
    pdf = gdf.to_pandas()
    gdf.loc[1] = 10
    pdf.loc[1] = 10
    assert_eq(gdf, pdf)


@pytest.mark.parametrize("typ", ["datetime64[ns]", "timedelta64[ns]"])
@pytest.mark.parametrize("idx_method, key", [["iloc", 0], ["loc", "a"]])
def test_series_iloc_scalar_datetimelike_return_pd_scalar(
    typ, idx_method, key
):
    obj = cudf.Series([1, 2, 3], index=list("abc"), dtype=typ)
    with cudf.option_context("mode.pandas_compatible", True):
        result = getattr(obj, idx_method)[key]
    expected = getattr(obj.to_pandas(), idx_method)[key]
    assert result == expected


@pytest.mark.parametrize("typ", ["datetime64[ns]", "timedelta64[ns]"])
@pytest.mark.parametrize(
    "idx_method, row_key, col_key", [["iloc", 0, 0], ["loc", "a", "a"]]
)
def test_dataframe_iloc_scalar_datetimelike_return_pd_scalar(
    typ, idx_method, row_key, col_key
):
    obj = cudf.DataFrame(
        [1, 2, 3], index=list("abc"), columns=["a"], dtype=typ
    )
    with cudf.option_context("mode.pandas_compatible", True):
        result = getattr(obj, idx_method)[row_key, col_key]
    expected = getattr(obj.to_pandas(), idx_method)[row_key, col_key]
    assert result == expected


@pytest.mark.parametrize("idx_method, key", [["iloc", 0], ["loc", "a"]])
def test_series_iloc_scalar_interval_return_pd_scalar(idx_method, key):
    iidx = cudf.IntervalIndex.from_breaks([1, 2, 3])
    obj = cudf.Series(iidx, index=list("ab"))
    with cudf.option_context("mode.pandas_compatible", True):
        result = getattr(obj, idx_method)[key]
    expected = getattr(obj.to_pandas(), idx_method)[key]
    assert result == expected


@pytest.mark.parametrize(
    "idx_method, row_key, col_key", [["iloc", 0, 0], ["loc", "a", "a"]]
)
def test_dataframe_iloc_scalar_interval_return_pd_scalar(
    idx_method, row_key, col_key
):
    iidx = cudf.IntervalIndex.from_breaks([1, 2, 3])
    obj = cudf.DataFrame({"a": iidx}, index=list("ab"))
    with cudf.option_context("mode.pandas_compatible", True):
        result = getattr(obj, idx_method)[row_key, col_key]
    expected = getattr(obj.to_pandas(), idx_method)[row_key, col_key]
    assert result == expected


def test_scalar_loc_row_categoricalindex():
    df = cudf.DataFrame(
        range(4), index=cudf.CategoricalIndex(["a", "a", "b", "c"])
    )
    result = df.loc["a"]
    expected = df.to_pandas().loc["a"]
    assert_eq(result, expected)


@pytest.mark.parametrize("klass", [cudf.DataFrame, cudf.Series])
@pytest.mark.parametrize("indexer", ["iloc", "loc"])
def test_iloc_loc_no_circular_reference(klass, indexer):
    obj = klass([0])
    ref = weakref.ref(obj)
    getattr(obj, indexer)[0]
    del obj
    assert ref() is None


def test_loc_setitem_empty_dataframe():
    pdf = pd.DataFrame(index=["index_1", "index_2", "index_3"])
    gdf = cudf.from_pandas(pdf)
    pdf.loc[["index_1"], "new_col"] = "A"
    gdf.loc[["index_1"], "new_col"] = "A"

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data",
    [
        [15, 14, 12, 10, 1],
        [1, 10, 12, 14, 15],
    ],
)
@pytest.mark.parametrize(
    "scalar",
    [
        1,
        10,
        15,
        14,
        0,
        2,
    ],
)
def test_loc_datetime_monotonic_with_ts(data, scalar):
    gdf = cudf.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5]},
        index=cudf.Index(data, dtype="datetime64[ns]"),
    )
    pdf = gdf.to_pandas()

    i = pd.Timestamp(scalar)

    actual = gdf.loc[i:]
    expected = pdf.loc[i:]

    assert_eq(actual, expected)

    actual = gdf.loc[:i]
    expected = pdf.loc[:i]

    assert_eq(actual, expected)


@pytest.mark.parametrize("data", [[15, 14, 3, 10, 1]])
@pytest.mark.parametrize("scalar", [1, 10, 15, 14, 0, 2])
def test_loc_datetime_random_with_ts(data, scalar):
    gdf = cudf.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5]},
        index=cudf.Index(data, dtype="datetime64[ns]"),
    )
    pdf = gdf.to_pandas()

    i = pd.Timestamp(scalar)

    if i not in pdf.index:
        assert_exceptions_equal(
            lambda: pdf.loc[i:],
            lambda: gdf.loc[i:],
            lfunc_args_and_kwargs=([],),
            rfunc_args_and_kwargs=([],),
        )
        assert_exceptions_equal(
            lambda: pdf.loc[:i],
            lambda: gdf.loc[:i],
            lfunc_args_and_kwargs=([],),
            rfunc_args_and_kwargs=([],),
        )
    else:
        actual = gdf.loc[i:]
        expected = pdf.loc[i:]

        assert_eq(actual, expected)

        actual = gdf.loc[:i]
        expected = pdf.loc[:i]

        assert_eq(actual, expected)

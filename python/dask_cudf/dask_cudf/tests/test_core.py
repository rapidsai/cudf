import random

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import dask
from dask import dataframe as dd
from dask.dataframe.core import make_meta, meta_nonempty

import cudf

import dask_cudf as dgd


def test_from_cudf():
    np.random.seed(0)

    df = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=10000),
            "y": np.random.normal(size=10000),
        }
    )

    gdf = cudf.DataFrame.from_pandas(df)

    # Test simple around to/from dask
    ingested = dd.from_pandas(gdf, npartitions=2)
    dd.assert_eq(ingested, df)

    # Test conversion to dask.dataframe
    ddf = ingested.to_dask_dataframe()
    dd.assert_eq(ddf, df)


def test_from_cudf_multiindex_raises():

    df = cudf.DataFrame({"x": list("abc"), "y": [1, 2, 3], "z": [1, 2, 3]})

    with pytest.raises(NotImplementedError):
        # dask_cudf does not support MultiIndex yet
        dgd.from_cudf(df.set_index(["x", "y"]))


def test_from_cudf_with_generic_idx():

    cdf = cudf.DataFrame(
        {
            "a": list(range(20)),
            "b": list(reversed(range(20))),
            "c": list(range(20)),
        }
    )

    ddf = dgd.from_cudf(cdf, npartitions=2)

    assert isinstance(ddf.index.compute(), cudf.core.index.GenericIndex)
    dd.assert_eq(ddf.loc[1:2, ["a"]], cdf.loc[1:2, ["a"]])


def _fragmented_gdf(df, nsplit):
    n = len(df)

    # Split dataframe in *nsplit*
    subdivsize = n // nsplit
    starts = [i * subdivsize for i in range(nsplit)]
    ends = starts[1:] + [None]
    frags = [df[s:e] for s, e in zip(starts, ends)]
    return frags


def test_query():
    np.random.seed(0)

    df = pd.DataFrame(
        {"x": np.random.randint(0, 5, size=10), "y": np.random.normal(size=10)}
    )
    gdf = cudf.DataFrame.from_pandas(df)
    expr = "x > 2"

    dd.assert_eq(gdf.query(expr), df.query(expr))

    queried = dd.from_pandas(gdf, npartitions=2).query(expr)

    got = queried
    expect = gdf.query(expr)

    dd.assert_eq(got, expect)


def test_query_local_dict():
    np.random.seed(0)
    df = pd.DataFrame(
        {"x": np.random.randint(0, 5, size=10), "y": np.random.normal(size=10)}
    )
    gdf = cudf.DataFrame.from_pandas(df)
    ddf = dgd.from_cudf(gdf, npartitions=2)

    val = 2

    gdf_queried = gdf.query("x > @val")
    ddf_queried = ddf.query("x > @val", local_dict={"val": val})

    dd.assert_eq(gdf_queried, ddf_queried)


def test_head():
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=100),
            "y": np.random.normal(size=100),
        }
    )
    gdf = cudf.DataFrame.from_pandas(df)
    dgf = dd.from_pandas(gdf, npartitions=2)

    dd.assert_eq(dgf.head(), df.head())


def test_from_dask_dataframe():
    np.random.seed(0)
    df = pd.DataFrame(
        {"x": np.random.randint(0, 5, size=20), "y": np.random.normal(size=20)}
    )
    ddf = dd.from_pandas(df, npartitions=2)
    dgdf = ddf.map_partitions(cudf.from_pandas)
    got = dgdf.compute().to_pandas()
    expect = df

    dd.assert_eq(got, expect)


@pytest.mark.parametrize("nelem", [10, 200, 1333])
@pytest.mark.parametrize("divisions", [None, "quantile"])
def test_set_index(nelem, divisions):
    with dask.config.set(scheduler="single-threaded"):
        np.random.seed(0)
        # Use unique index range as the sort may not be stable-ordering
        x = np.arange(nelem)
        np.random.shuffle(x)
        df = pd.DataFrame(
            {"x": x, "y": np.random.randint(0, nelem, size=nelem)}
        )
        ddf = dd.from_pandas(df, npartitions=2)
        dgdf = ddf.map_partitions(cudf.from_pandas)

        expect = ddf.set_index("x")
        got = dgdf.set_index("x", divisions=divisions)

        dd.assert_eq(expect, got, check_index=False, check_divisions=False)


@pytest.mark.parametrize("by", ["a", "b"])
@pytest.mark.parametrize("nelem", [10, 500])
@pytest.mark.parametrize("nparts", [1, 10])
def test_set_index_quantile(nelem, nparts, by):
    df = cudf.DataFrame()
    df["a"] = np.ascontiguousarray(np.arange(nelem)[::-1])
    df["b"] = np.random.choice(cudf.datasets.names, size=nelem)
    ddf = dd.from_pandas(df, npartitions=nparts)

    got = ddf.set_index(by, divisions="quantile")
    expect = df.sort_values(by=by).set_index(by)
    dd.assert_eq(got, expect)


def assert_frame_equal_by_index_group(expect, got):
    assert sorted(expect.columns) == sorted(got.columns)
    assert sorted(set(got.index)) == sorted(set(expect.index))
    # Note the set_index sort is not stable,
    unique_values = sorted(set(got.index))
    for iv in unique_values:
        sr_expect = expect.loc[[iv]]
        sr_got = got.loc[[iv]]

        for k in expect.columns:
            # Sort each column before we compare them
            sorted_expect = sr_expect.sort_values(k)[k]
            sorted_got = sr_got.sort_values(k)[k]
            np.testing.assert_array_equal(sorted_expect, sorted_got)


@pytest.mark.parametrize("nelem", [10, 200, 1333])
def test_set_index_2(nelem):
    with dask.config.set(scheduler="single-threaded"):
        np.random.seed(0)
        df = pd.DataFrame(
            {
                "x": 100 + np.random.randint(0, nelem // 2, size=nelem),
                "y": np.random.normal(size=nelem),
            }
        )
        expect = df.set_index("x").sort_index()

        dgf = dd.from_pandas(cudf.DataFrame.from_pandas(df), npartitions=4)
        res = dgf.set_index("x")  # sort by default
        got = res.compute().to_pandas()

        assert_frame_equal_by_index_group(expect, got)


@pytest.mark.xfail(reason="dask's index name '__dask_cudf.index' is correct")
def test_set_index_w_series():
    with dask.config.set(scheduler="single-threaded"):
        nelem = 20
        np.random.seed(0)
        df = pd.DataFrame(
            {
                "x": 100 + np.random.randint(0, nelem // 2, size=nelem),
                "y": np.random.normal(size=nelem),
            }
        )
        expect = df.set_index(df.x).sort_index()

        dgf = dd.from_pandas(cudf.DataFrame.from_pandas(df), npartitions=4)
        res = dgf.set_index(dgf.x)  # sort by default
        got = res.compute().to_pandas()

        dd.assert_eq(expect, got)


def test_set_index_sorted():
    with dask.config.set(scheduler="single-threaded"):
        df1 = pd.DataFrame({"val": [4, 3, 2, 1, 0], "id": [0, 1, 3, 5, 7]})
        ddf1 = dd.from_pandas(df1, npartitions=2)

        gdf1 = cudf.from_pandas(df1)
        gddf1 = dgd.from_cudf(gdf1, npartitions=2)

        expect = ddf1.set_index("id", sorted=True)
        got = gddf1.set_index("id", sorted=True)

        dd.assert_eq(expect, got)

        with pytest.raises(ValueError):
            # Cannot set `sorted=True` for non-sorted column
            gddf1.set_index("val", sorted=True)


@pytest.mark.parametrize("nelem", [10, 200, 1333])
@pytest.mark.parametrize("index", [None, "myindex"])
def test_rearrange_by_divisions(nelem, index):
    with dask.config.set(scheduler="single-threaded"):
        np.random.seed(0)
        df = pd.DataFrame(
            {
                "x": np.random.randint(0, 20, size=nelem),
                "y": np.random.normal(size=nelem),
                "z": np.random.choice(["dog", "cat", "bird"], nelem),
            }
        )
        df["z"] = df["z"].astype("category")

        ddf1 = dd.from_pandas(df, npartitions=4)
        gdf1 = dgd.from_cudf(cudf.DataFrame.from_pandas(df), npartitions=4)
        ddf1.index.name = index
        gdf1.index.name = index
        divisions = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)

        expect = dd.shuffle.rearrange_by_divisions(
            ddf1, "x", divisions=divisions, shuffle="tasks"
        )
        result = dd.shuffle.rearrange_by_divisions(
            gdf1, "x", divisions=divisions, shuffle="tasks"
        )
        dd.assert_eq(expect, result)


def test_assign():
    np.random.seed(0)
    df = pd.DataFrame(
        {"x": np.random.randint(0, 5, size=20), "y": np.random.normal(size=20)}
    )

    dgf = dd.from_pandas(cudf.DataFrame.from_pandas(df), npartitions=2)
    pdcol = pd.Series(np.arange(20) + 1000)
    newcol = dd.from_pandas(cudf.Series(pdcol), npartitions=dgf.npartitions)
    got = dgf.assign(z=newcol)

    dd.assert_eq(got.loc[:, ["x", "y"]], df)
    np.testing.assert_array_equal(got["z"].compute().to_array(), pdcol)


@pytest.mark.parametrize("data_type", ["int8", "int16", "int32", "int64"])
def test_setitem_scalar_integer(data_type):
    np.random.seed(0)
    scalar = np.random.randint(0, 100, dtype=data_type)
    df = pd.DataFrame(
        {"x": np.random.randint(0, 5, size=20), "y": np.random.normal(size=20)}
    )
    dgf = dd.from_pandas(cudf.DataFrame.from_pandas(df), npartitions=2)

    df["z"] = scalar
    dgf["z"] = scalar

    got = dgf.compute().to_pandas()
    np.testing.assert_array_equal(got["z"], df["z"])


@pytest.mark.parametrize("data_type", ["float32", "float64"])
def test_setitem_scalar_float(data_type):
    np.random.seed(0)
    scalar = np.random.randn(1).astype(data_type)[0]
    df = pd.DataFrame(
        {"x": np.random.randint(0, 5, size=20), "y": np.random.normal(size=20)}
    )
    dgf = dd.from_pandas(cudf.DataFrame.from_pandas(df), npartitions=2)

    df["z"] = scalar
    dgf["z"] = scalar

    got = dgf.compute().to_pandas()
    np.testing.assert_array_equal(got["z"], df["z"])


def test_setitem_scalar_datetime():
    np.random.seed(0)
    scalar = np.int64(np.random.randint(0, 100)).astype("datetime64[ms]")
    df = pd.DataFrame(
        {"x": np.random.randint(0, 5, size=20), "y": np.random.normal(size=20)}
    )
    dgf = dd.from_pandas(cudf.DataFrame.from_pandas(df), npartitions=2)

    df["z"] = scalar
    dgf["z"] = scalar

    got = dgf.compute().to_pandas()
    np.testing.assert_array_equal(got["z"], df["z"])


@pytest.mark.parametrize(
    "func",
    [
        lambda: pd._testing.makeDataFrame().reset_index(),
        pd._testing.makeDataFrame,
        pd._testing.makeMixedDataFrame,
        pd._testing.makeObjectSeries,
        pd._testing.makeTimeSeries,
    ],
)
def test_repr(func):
    pdf = func()
    try:
        gdf = cudf.from_pandas(pdf)
    except Exception:
        raise pytest.xfail()
    # gddf = dd.from_pandas(gdf, npartitions=3, sort=False)  # TODO
    gddf = dd.from_pandas(gdf, npartitions=3, sort=False)

    assert repr(gddf)
    if hasattr(pdf, "_repr_html_"):
        assert gddf._repr_html_()


@pytest.mark.skip(reason="datetime indexes not fully supported in cudf")
@pytest.mark.parametrize("start", ["1d", "5d", "1w", "12h"])
@pytest.mark.parametrize("stop", ["1d", "3d", "8h"])
def test_repartition_timeseries(start, stop):
    # This test is currently absurdly slow.  It should not be unskipped without
    # slimming it down.
    pdf = dask.datasets.timeseries(
        "2000-01-01",
        "2000-01-31",
        freq="1s",
        partition_freq=start,
        dtypes={"x": int, "y": float},
    )
    gdf = pdf.map_partitions(cudf.DataFrame.from_pandas)

    a = pdf.repartition(freq=stop)
    b = gdf.repartition(freq=stop)
    assert a.divisions == b.divisions

    dd.utils.assert_eq(a, b)


@pytest.mark.parametrize("start", [1, 2, 5])
@pytest.mark.parametrize("stop", [1, 3, 7])
def test_repartition_simple_divisions(start, stop):
    pdf = pd.DataFrame({"x": range(100)})

    pdf = dd.from_pandas(pdf, npartitions=start)
    gdf = pdf.map_partitions(cudf.DataFrame.from_pandas)

    a = pdf.repartition(npartitions=stop)
    b = gdf.repartition(npartitions=stop)
    assert a.divisions == b.divisions

    dd.assert_eq(a, b)


@pytest.mark.parametrize("npartitions", [2, 17, 20])
def test_repartition_hash_staged(npartitions):
    by = ["b"]
    datarange = 35
    size = 100
    gdf = cudf.DataFrame(
        {
            "a": np.arange(size, dtype="int64"),
            "b": np.random.randint(datarange, size=size),
        }
    )
    # WARNING: Specific npartitions-max_branch combination
    # was specifically chosen to cover changes in #4676
    npartitions_initial = 17
    ddf = dgd.from_cudf(gdf, npartitions=npartitions_initial)
    ddf_new = ddf.repartition(
        columns=by, npartitions=npartitions, max_branch=4
    )

    # Make sure we are getting a dask_cudf dataframe
    assert type(ddf_new) == type(ddf)

    # Check that the length was preserved
    assert len(ddf_new) == len(ddf)

    # Check that the partitions have unique keys,
    # and that the key values are preserved
    expect_unique = gdf[by].drop_duplicates().sort_values(by)
    got_unique = cudf.concat(
        [
            part[by].compute().drop_duplicates()
            for part in ddf_new[by].partitions
        ],
        ignore_index=True,
    ).sort_values(by)
    dd.assert_eq(got_unique, expect_unique, check_index=False)


@pytest.mark.parametrize("by", [["b"], ["c"], ["d"], ["b", "c"]])
@pytest.mark.parametrize("npartitions", [3, 4, 5])
@pytest.mark.parametrize("max_branch", [3, 32])
def test_repartition_hash(by, npartitions, max_branch):
    npartitions_i = 4
    datarange = 26
    size = 100
    gdf = cudf.DataFrame(
        {
            "a": np.arange(0, stop=size, dtype="int64"),
            "b": np.random.randint(datarange, size=size),
            "c": np.random.choice(list("abcdefgh"), size=size),
            "d": np.random.choice(np.arange(26), size=size),
        }
    )
    gdf.d = gdf.d.astype("datetime64[ms]")
    ddf = dgd.from_cudf(gdf, npartitions=npartitions_i)
    ddf_new = ddf.repartition(
        columns=by, npartitions=npartitions, max_branch=max_branch
    )

    # Check that the length was preserved
    assert len(ddf_new) == len(ddf)

    # Check that the partitions have unique keys,
    # and that the key values are preserved
    expect_unique = gdf[by].drop_duplicates().sort_values(by)
    got_unique = cudf.concat(
        [
            part[by].compute().drop_duplicates()
            for part in ddf_new[by].partitions
        ],
        ignore_index=True,
    ).sort_values(by)
    dd.assert_eq(got_unique, expect_unique, check_index=False)


@pytest.fixture
def pdf():
    return pd.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "y": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]}
    )


@pytest.fixture
def gdf(pdf):
    return cudf.from_pandas(pdf)


@pytest.fixture
def ddf(pdf):
    return dd.from_pandas(pdf, npartitions=3)


@pytest.fixture
def gddf(gdf):
    return dd.from_pandas(gdf, npartitions=3)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df + 1,
        lambda df: df.index,
        lambda df: df.x.sum(),
        lambda df: df.x.astype(float),
        lambda df: df.assign(z=df.x.astype("int")),
    ],
)
def test_unary_ops(func, gdf, gddf):
    p = func(gdf)
    g = func(gddf)

    # Fixed in https://github.com/dask/dask/pull/4657
    if isinstance(p, cudf.Index):
        from packaging import version

        if version.parse(dask.__version__) < version.parse("1.1.6"):
            pytest.skip(
                "dask.dataframe assert_eq index check hardcoded to "
                "pandas prior to 1.1.6 release"
            )

    dd.assert_eq(p, g, check_names=False)


@pytest.mark.parametrize("series", [True, False])
def test_concat(gdf, gddf, series):
    if series:
        gdf = gdf.x
        gddf = gddf.x
        a = (
            cudf.concat([gdf, gdf + 1, gdf + 2])
            .sort_values()
            .reset_index(drop=True)
        )
        b = (
            dd.concat([gddf, gddf + 1, gddf + 2], interleave_partitions=True)
            .compute()
            .sort_values()
            .reset_index(drop=True)
        )
    else:
        a = (
            cudf.concat([gdf, gdf + 1, gdf + 2])
            .sort_values("x")
            .reset_index(drop=True)
        )
        b = (
            dd.concat([gddf, gddf + 1, gddf + 2], interleave_partitions=True)
            .compute()
            .sort_values("x")
            .reset_index(drop=True)
        )
    dd.assert_eq(a, b)


def test_boolean_index(gdf, gddf):

    gdf2 = gdf[gdf.x > 2]
    gddf2 = gddf[gddf.x > 2]

    dd.assert_eq(gdf2, gddf2)


def test_drop(gdf, gddf):
    gdf2 = gdf.drop(columns="x")
    gddf2 = gddf.drop(columns="x").compute()

    dd.assert_eq(gdf2, gddf2)


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize("index", [True, False])
def test_memory_usage(gdf, gddf, index, deep):
    dd.assert_eq(
        gdf.memory_usage(deep=deep, index=index),
        gddf.memory_usage(deep=deep, index=index),
    )


@pytest.mark.parametrize("index", [True, False])
def test_hash_object_dispatch(index):
    obj = cudf.DataFrame(
        {"x": ["a", "b", "c"], "y": [1, 2, 3], "z": [1, 1, 0]}, index=[2, 4, 6]
    )

    # DataFrame
    result = dd.utils.hash_object_dispatch(obj, index=index)
    expected = dgd.backends.hash_object_cudf(obj, index=index)
    assert isinstance(result, cudf.Series)
    dd.assert_eq(result, expected)

    # Series
    result = dd.utils.hash_object_dispatch(obj["x"], index=index)
    expected = dgd.backends.hash_object_cudf(obj["x"], index=index)
    assert isinstance(result, cudf.Series)
    dd.assert_eq(result, expected)

    # DataFrame with MultiIndex
    obj_multi = obj.set_index(["x", "z"], drop=True)
    result = dd.utils.hash_object_dispatch(obj_multi, index=index)
    expected = dgd.backends.hash_object_cudf(obj_multi, index=index)
    assert isinstance(result, cudf.Series)
    dd.assert_eq(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        "int8",
        "int32",
        "int64",
        "float64",
        "strings",
        "cats",
        "time_s",
        "time_ms",
        "time_ns",
        ["int32", "int64"],
        ["int8", "float64", "strings"],
        ["cats", "int8", "float64"],
        ["time_ms", "cats"],
    ],
)
def test_make_meta_backends(index):

    dtypes = ["int8", "int32", "int64", "float64"]
    df = cudf.DataFrame(
        {dt: np.arange(start=0, stop=3, dtype=dt) for dt in dtypes}
    )
    df["strings"] = ["cat", "dog", "fish"]
    df["cats"] = df["strings"].astype("category")
    df["time_s"] = np.array(
        ["2018-10-07", "2018-10-08", "2018-10-09"], dtype="datetime64[s]"
    )
    df["time_ms"] = df["time_s"].astype("datetime64[ms]")
    df["time_ns"] = df["time_s"].astype("datetime64[ns]")
    df = df.set_index(index)

    # Check "empty" metadata types
    chk_meta = make_meta(df)
    dd.assert_eq(chk_meta.dtypes, df.dtypes)

    # Check "non-empty" metadata types
    chk_meta_nonempty = meta_nonempty(df)
    dd.assert_eq(chk_meta.dtypes, chk_meta_nonempty.dtypes)

    # Check dask code path if not MultiIndex
    if not isinstance(df.index, cudf.MultiIndex):

        ddf = dgd.from_cudf(df, npartitions=1)

        # Check "empty" metadata types
        dd.assert_eq(ddf._meta.dtypes, df.dtypes)

        # Check "non-empty" metadata types
        dd.assert_eq(ddf._meta.dtypes, ddf._meta_nonempty.dtypes)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([]),
        pd.DataFrame({"abc": [], "xyz": []}),
        pd.Series([1, 2, 10, 11]),
        pd.DataFrame({"abc": [1, 2, 10, 11], "xyz": [100, 12, 120, 1]}),
    ],
)
def test_dataframe_series_replace(data):
    pdf = data.copy()
    gdf = cudf.from_pandas(pdf)

    ddf = dgd.from_cudf(gdf, npartitions=5)

    dd.assert_eq(ddf.replace(1, 2), pdf.replace(1, 2))


def test_dataframe_assign_col():
    df = cudf.DataFrame(list(range(100)))
    pdf = pd.DataFrame(list(range(100)))

    ddf = dgd.from_cudf(df, npartitions=4)
    ddf["fold"] = 0
    ddf["fold"] = ddf["fold"].map_partitions(
        lambda cudf_df: cp.random.randint(0, 4, len(cudf_df))
    )

    pddf = dd.from_pandas(pdf, npartitions=4)
    pddf["fold"] = 0
    pddf["fold"] = pddf["fold"].map_partitions(
        lambda p_df: np.random.randint(0, 4, len(p_df))
    )

    dd.assert_eq(ddf[0], pddf[0])
    dd.assert_eq(len(ddf["fold"]), len(pddf["fold"]))


def test_dataframe_set_index():
    random.seed(0)
    df = cudf.datasets.randomdata(26, dtypes={"a": float, "b": int})
    df["str"] = list("abcdefghijklmnopqrstuvwxyz")
    pdf = df.to_pandas()

    ddf = dgd.from_cudf(df, npartitions=4)
    ddf = ddf.set_index("str")

    pddf = dd.from_pandas(pdf, npartitions=4)
    pddf = pddf.set_index("str")
    from cudf.tests.utils import assert_eq

    assert_eq(ddf.compute(), pddf.compute())


def test_dataframe_describe():
    random.seed(0)
    df = cudf.datasets.randomdata(20)
    pdf = df.to_pandas()

    ddf = dgd.from_cudf(df, npartitions=4)
    pddf = dd.from_pandas(pdf, npartitions=4)

    dd.assert_eq(ddf.describe(), pddf.describe(), check_less_precise=3)

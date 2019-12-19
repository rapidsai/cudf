import numpy as np
import pandas as pd
import pandas.util.testing as tm
import pytest

import dask
import dask.dataframe as dd

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
def test_set_index(nelem):
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
        got = dgdf.set_index("x")

        dd.assert_eq(expect, got, check_index=False, check_divisions=False)


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
    out = dgf.assign(z=newcol)

    got = out
    dd.assert_eq(got.loc[:, ["x", "y"]], df)
    np.testing.assert_array_equal(got["z"], pdcol)


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
        lambda: tm.makeDataFrame().reset_index(),
        tm.makeDataFrame,
        tm.makeMixedDataFrame,
        tm.makeObjectSeries,
        tm.makeTimeSeries,
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

    dd.utils.assert_eq(a, b)


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

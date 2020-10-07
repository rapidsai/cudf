import numpy as np
import pandas as pd
import pytest

import dask
from dask import dataframe as dd

import dask_cudf

import cudf


@pytest.mark.parametrize("aggregation", ["sum", "mean", "count", "min", "max"])
def test_groupby_basic_aggs(aggregation):
    pdf = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=10000),
            "y": np.random.normal(size=10000),
        }
    )

    gdf = cudf.DataFrame.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = getattr(gdf.groupby("x"), aggregation)()
    b = getattr(ddf.groupby("x"), aggregation)().compute()

    if aggregation == "count":
        dd.assert_eq(a, b, check_dtype=False)
    else:
        dd.assert_eq(a, b)

    a = gdf.groupby("x").agg({"x": aggregation})
    b = ddf.groupby("x").agg({"x": aggregation}).compute()

    if aggregation == "count":
        dd.assert_eq(a, b, check_dtype=False)
    else:
        dd.assert_eq(a, b)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.groupby("x").agg({"y": "max"}),
        pytest.param(
            lambda df: df.groupby("x").y.agg(["sum", "max"]),
            marks=pytest.mark.skip,
        ),
    ],
)
def test_groupby_agg(func):
    pdf = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=10000),
            "y": np.random.normal(size=10000),
        }
    )

    gdf = cudf.DataFrame.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = func(gdf).to_pandas()
    b = func(ddf).compute().to_pandas()

    a.index.name = None
    a.name = None
    b.index.name = None
    b.name = None

    dd.assert_eq(a, b)


@pytest.mark.parametrize("split_out", [1, 3])
def test_groupby_agg_empty_partition(tmpdir, split_out):

    # Write random and empty cudf DataFrames
    # to two distinct files.
    df = cudf.datasets.randomdata()
    df.to_parquet(str(tmpdir.join("f0.parquet")))
    cudf.DataFrame(
        columns=["id", "x", "y"],
        dtype={"id": "int64", "x": "float64", "y": "float64"},
    ).to_parquet(str(tmpdir.join("f1.parquet")))

    # Read back our two partitions as a single
    # dask_cudf DataFrame (one partition is now empty)
    ddf = dask_cudf.read_parquet(str(tmpdir))
    gb = ddf.groupby(["id"]).agg({"x": ["sum"]}, split_out=split_out)

    expect = df.groupby(["id"]).agg({"x": ["sum"]}).sort_index()
    dd.assert_eq(gb.compute().sort_index(), expect)


@pytest.mark.xfail(reason="cudf issues")
@pytest.mark.parametrize(
    "func",
    [lambda df: df.groupby("x").std(), lambda df: df.groupby("x").y.std()],
)
def test_groupby_std(func):
    pdf = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=10000),
            "y": np.random.normal(size=10000),
        }
    )

    gdf = cudf.DataFrame.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = func(gdf.to_pandas())
    b = func(ddf).compute().to_pandas()

    a.index.name = None
    a.name = None
    b.index.name = None

    dd.assert_eq(a, b)


# reason gotattr in cudf
@pytest.mark.parametrize(
    "func",
    [
        pytest.param(
            lambda df: df.groupby(["a", "b"]).x.sum(), marks=pytest.mark.xfail
        ),
        pytest.param(
            lambda df: df.groupby(["a", "b"]).sum(), marks=pytest.mark.xfail
        ),
        pytest.param(
            lambda df: df.groupby(["a", "b"]).agg({"x", "sum"}),
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_groupby_multi_column(func):
    pdf = pd.DataFrame(
        {
            "a": np.random.randint(0, 20, size=1000),
            "b": np.random.randint(0, 5, size=1000),
            "x": np.random.normal(size=1000),
        }
    )

    gdf = cudf.DataFrame.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = func(gdf).to_pandas()
    b = func(ddf).compute().to_pandas()

    dd.assert_eq(a, b)


def test_reset_index_multiindex():
    df = cudf.DataFrame()
    df["id_1"] = ["a", "a", "b"]
    df["id_2"] = [0, 0, 1]
    df["val"] = [1, 2, 3]

    df_lookup = cudf.DataFrame()
    df_lookup["id_1"] = ["a", "b"]
    df_lookup["metadata"] = [0, 1]

    gddf = dask_cudf.from_cudf(df, npartitions=2)
    gddf_lookup = dask_cudf.from_cudf(df_lookup, npartitions=2)

    ddf = dd.from_pandas(df.to_pandas(), npartitions=2)
    ddf_lookup = dd.from_pandas(df_lookup.to_pandas(), npartitions=2)

    # Note: 'id_2' has wrong type (object) until after compute
    dd.assert_eq(
        gddf.groupby(by=["id_1", "id_2"])
        .val.sum()
        .reset_index()
        .merge(gddf_lookup, on="id_1")
        .compute(),
        ddf.groupby(by=["id_1", "id_2"])
        .val.sum()
        .reset_index()
        .merge(ddf_lookup, on="id_1"),
    )


@pytest.mark.parametrize("split_out", [1, 2, 3])
@pytest.mark.parametrize(
    "column", ["c", "d", "e", ["b", "c"], ["b", "d"], ["b", "e"]]
)
def test_groupby_split_out(split_out, column):
    df = pd.DataFrame(
        {
            "a": np.arange(8),
            "b": [1, 0, 0, 2, 1, 1, 2, 0],
            "c": [0, 1] * 4,
            "d": ["dog", "cat", "cat", "dog", "dog", "dog", "cat", "bird"],
        }
    ).fillna(0)
    df["e"] = df["d"].astype("category")

    gdf = cudf.from_pandas(df)

    ddf = dd.from_pandas(df, npartitions=3)
    gddf = dask_cudf.from_cudf(gdf, npartitions=3)

    ddf_result = (
        ddf.groupby(column)
        .a.mean(split_out=split_out)
        .compute()
        .sort_values()
        .dropna()
    )
    gddf_result = (
        gddf.groupby(column)
        .a.mean(split_out=split_out)
        .compute()
        .sort_values()
    )

    dd.assert_eq(gddf_result, ddf_result, check_index=False)


@pytest.mark.parametrize("dropna", [False, True, None])
@pytest.mark.parametrize(
    "by", ["a", "b", "c", "d", ["a", "b"], ["a", "c"], ["a", "d"]]
)
def test_groupby_dropna(dropna, by):

    # NOTE: This test is borrowed from upstream dask
    #       (dask/dask/dataframe/tests/test_groupby.py)
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, None, None, 7, 8],
            "b": [1, None, 1, 3, None, 3, 1, 3],
            "c": ["a", "b", None, None, "e", "f", "g", "h"],
            "e": [4, 5, 6, 3, 2, 1, 0, 0],
        }
    )
    df["b"] = df["b"].astype("datetime64[ns]")
    df["d"] = df["c"].astype("category")
    ddf = dask_cudf.from_cudf(df, npartitions=3)

    if dropna is None:
        dask_result = ddf.groupby(by).e.sum()
        cudf_result = df.groupby(by).e.sum()
    else:
        dask_result = ddf.groupby(by, dropna=dropna).e.sum()
        cudf_result = df.groupby(by, dropna=dropna).e.sum()
    if by in ["c", "d"]:
        # Loose string/category index name in cudf...
        dask_result = dask_result.compute()
        dask_result.index.name = cudf_result.index.name

    dd.assert_eq(dask_result, cudf_result)


@pytest.mark.parametrize("myindex", [[1, 2] * 4, ["s1", "s2"] * 4])
def test_groupby_string_index_name(myindex):
    # GH-Issue #3420
    data = {"index": myindex, "data": [0, 1] * 4}
    df = cudf.DataFrame(data=data)
    ddf = dask_cudf.from_cudf(df, npartitions=2)
    gdf = ddf.groupby("index").agg({"data": "count"})

    assert gdf.compute().index.name == gdf.index.name


@pytest.mark.parametrize(
    "agg_func",
    [
        lambda gb: gb.agg({"c": ["count"]}, split_out=2),
        lambda gb: gb.agg({"c": "count"}, split_out=2),
        lambda gb: gb.agg({"c": ["count", "sum"]}, split_out=2),
        lambda gb: gb.count(split_out=2),
        lambda gb: gb.c.count(split_out=2),
    ],
)
def test_groupby_split_out_multiindex(agg_func):
    df = cudf.DataFrame(
        {
            "a": np.random.randint(0, 10, 100),
            "b": np.random.randint(0, 5, 100),
            "c": np.random.random(100),
        }
    )
    ddf = dask_cudf.from_cudf(df, 5)
    pddf = dd.from_pandas(df.to_pandas(), 5)
    gr = agg_func(ddf.groupby(["a", "b"]))
    pr = agg_func(pddf.groupby(["a", "b"]))
    dd.assert_eq(gr.compute(), pr.compute())


@pytest.mark.parametrize("npartitions", [1, 2])
def test_groupby_multiindex_reset_index(npartitions):
    df = cudf.DataFrame(
        {"a": [1, 1, 2, 3, 4], "b": [5, 2, 1, 2, 5], "c": [1, 2, 2, 3, 5]}
    )
    ddf = dask_cudf.from_cudf(df, npartitions=npartitions)
    pddf = dd.from_pandas(df.to_pandas(), npartitions=npartitions)
    gr = ddf.groupby(["a", "c"]).agg({"b": ["count"]}).reset_index()
    pr = pddf.groupby(["a", "c"]).agg({"b": ["count"]}).reset_index()

    # CuDF uses "int32" for count. Pandas uses "int64"
    gr_out = gr.compute().sort_values(by=["a", "c"]).reset_index(drop=True)
    gr_out[("b", "count")] = gr_out[("b", "count")].astype("int64")

    dd.assert_eq(
        gr_out, pr.compute().sort_values(by=["a", "c"]).reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "groupby_keys", [["a"], ["a", "b"], ["a", "b", "dd"], ["a", "dd", "b"]]
)
@pytest.mark.parametrize(
    "agg_func",
    [
        lambda gb: gb.agg({"c": ["count"]}),
        lambda gb: gb.agg({"c": "count"}),
        lambda gb: gb.agg({"c": ["count", "sum"]}),
        lambda gb: gb.count(),
        lambda gb: gb.c.count(),
    ],
)
def test_groupby_reset_index_multiindex(groupby_keys, agg_func):
    df = cudf.DataFrame(
        {
            "a": np.random.randint(0, 10, 10),
            "b": np.random.randint(0, 5, 10),
            "c": np.random.randint(0, 5, 10),
            "dd": np.random.randint(0, 5, 10),
        }
    )
    ddf = dask_cudf.from_cudf(df, 5)
    pddf = dd.from_pandas(df.to_pandas(), 5)
    gr = agg_func(ddf.groupby(groupby_keys)).reset_index()
    pr = agg_func(pddf.groupby(groupby_keys)).reset_index()
    gf = gr.compute().sort_values(groupby_keys).reset_index(drop=True)
    pf = pr.compute().sort_values(groupby_keys).reset_index(drop=True)
    dd.assert_eq(gf, pf)


def test_groupby_reset_index_drop_True():
    df = cudf.DataFrame(
        {"a": np.random.randint(0, 10, 10), "b": np.random.randint(0, 5, 10)}
    )
    ddf = dask_cudf.from_cudf(df, 5)
    pddf = dd.from_pandas(df.to_pandas(), 5)
    gr = ddf.groupby(["a"]).agg({"b": ["count"]}).reset_index(drop=True)
    pr = pddf.groupby(["a"]).agg({"b": ["count"]}).reset_index(drop=True)
    gf = gr.compute().sort_values(by=["b"]).reset_index(drop=True)
    pf = pr.compute().sort_values(by=[("b", "count")]).reset_index(drop=True)
    dd.assert_eq(gf, pf)


def test_groupby_mean_sort_false():
    df = cudf.datasets.randomdata(nrows=150, dtypes={"a": int, "b": int})
    ddf = dask_cudf.from_cudf(df, 1)
    pddf = dd.from_pandas(df.to_pandas(), 1)

    gr = ddf.groupby(["a"]).agg({"b": "mean"})
    pr = pddf.groupby(["a"]).agg({"b": "mean"})
    assert pr.index.name == gr.index.name
    assert pr.head(0).index.name == gr.head(0).index.name

    gf = gr.compute().sort_values(by=["b"]).reset_index(drop=True)
    pf = pr.compute().sort_values(by=["b"]).reset_index(drop=True)
    dd.assert_eq(gf, pf)


def test_groupby_reset_index_dtype():

    # Make sure int8 dtype is properly preserved
    # Through various cudf/dask_cudf ops
    #
    # Note: GitHub Issue#4090 reproducer

    df = cudf.DataFrame()
    df["a"] = np.arange(10, dtype="int8")
    df["b"] = np.arange(10, dtype="int8")
    df = dask_cudf.from_cudf(df, 1)

    a = df.groupby("a").agg({"b": ["count"]})

    assert a.index.dtype == "int8"
    assert a.reset_index().dtypes[0] == "int8"


def test_groupby_reset_index_names():
    df = cudf.datasets.randomdata(
        nrows=10, dtypes={"a": str, "b": int, "c": int}
    )
    pdf = df.to_pandas()

    gddf = dask_cudf.from_cudf(df, 2)
    pddf = dd.from_pandas(pdf, 2)

    g_res = gddf.groupby("a", sort=True).sum()
    p_res = pddf.groupby("a", sort=True).sum()

    got = g_res.reset_index().compute().sort_values(["a", "b", "c"])
    expect = p_res.reset_index().compute().sort_values(["a", "b", "c"])

    dd.assert_eq(got, expect)


def test_groupby_reset_index_string_name():
    df = cudf.DataFrame({"value": range(5), "key": ["a", "a", "b", "a", "c"]})
    pdf = df.to_pandas()

    gddf = dask_cudf.from_cudf(df, npartitions=1)
    pddf = dd.from_pandas(pdf, npartitions=1)

    g_res = (
        gddf.groupby(["key"]).agg({"value": "mean"}).reset_index(drop=False)
    )
    p_res = (
        pddf.groupby(["key"]).agg({"value": "mean"}).reset_index(drop=False)
    )

    got = g_res.compute().sort_values(["key", "value"]).reset_index(drop=True)
    expect = (
        p_res.compute().sort_values(["key", "value"]).reset_index(drop=True)
    )

    dd.assert_eq(got, expect)
    assert len(g_res) == len(p_res)


def test_groupby_categorical_key():
    # See https://github.com/rapidsai/cudf/issues/4608
    df = dask.datasets.timeseries()
    gddf = dask_cudf.from_dask_dataframe(df)
    gddf["name"] = gddf["name"].astype("category")
    ddf = gddf.to_dask_dataframe()

    got = (
        gddf.groupby("name")
        .agg({"x": ["mean", "max"], "y": ["mean", "count"]})
        .compute()
    )
    expect = (
        ddf.groupby("name")
        .agg({"x": ["mean", "max"], "y": ["mean", "count"]})
        .compute()
    )
    dd.assert_eq(expect, got)


@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("split_out", [None, 1, 2])
@pytest.mark.parametrize("split_every", [False, 4])
@pytest.mark.parametrize("npartitions", [1, 10])
def test_groupby_agg_params(npartitions, split_every, split_out, as_index):
    df = cudf.datasets.randomdata(
        nrows=150, dtypes={"name": str, "a": int, "b": int, "c": float},
    )
    df["a"] = [0, 1, 2] * 50
    ddf = dask_cudf.from_cudf(df, npartitions)
    pddf = dd.from_pandas(df.to_pandas(), npartitions)

    agg_dict = {
        "a": "sum",
        "b": ["min", "max", "mean"],
        "c": ["mean", "std", "var"],
    }

    # Check `sort=True` behavior
    if split_out == 1:
        gf = (
            ddf.groupby(["name", "a"], sort=True, as_index=as_index)
            .aggregate(agg_dict, split_every=split_every, split_out=split_out,)
            .compute()
        )
        if as_index:
            # Groupby columns became the index.
            # Sorting the index should not change anything.
            dd.assert_eq(gf.index, gf.sort_index().index)
        else:
            # Groupby columns are did NOT become the index.
            # Sorting by these columns should not change anything.
            sort_cols = [("name", ""), ("a", "")]
            dd.assert_eq(
                gf[sort_cols],
                gf[sort_cols].sort_values(sort_cols),
                check_index=False,
            )

    # Full check (`sort=False`)
    gr = ddf.groupby(["name", "a"], sort=False, as_index=as_index).aggregate(
        agg_dict, split_every=split_every, split_out=split_out,
    )
    pr = pddf.groupby(["name", "a"], sort=False).agg(
        agg_dict, split_every=split_every, split_out=split_out,
    )

    # Test `as_index` argument
    if as_index:
        # Groupby columns should NOT be in columns
        assert ("name", "") not in gr.columns and ("a", "") not in gr.columns
    else:
        # Groupby columns SHOULD be in columns
        assert ("name", "") in gr.columns and ("a", "") in gr.columns

    # Check `split_out` argument
    assert gr.npartitions == (split_out or 1)

    # Compute for easier multiindex handling
    gf = gr.compute()
    pf = pr.compute()

    # Reset index and sort by groupby columns
    if as_index:
        gf = gf.reset_index(drop=False)
    sort_cols = [("name", ""), ("a", ""), ("c", "mean")]
    gf = gf.sort_values(sort_cols).reset_index(drop=True)
    pf = (
        pf.reset_index(drop=False)
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )

    dd.assert_eq(gf, pf)

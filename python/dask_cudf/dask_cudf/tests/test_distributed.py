# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import numba.cuda
import pytest

import dask
from dask import array as da, dataframe as dd
from dask.distributed import Client
from distributed.utils_test import cleanup, loop, loop_in_thread  # noqa: F401

import cudf
from cudf.testing import assert_eq

import dask_cudf

dask_cuda = pytest.importorskip("dask_cuda")


def at_least_n_gpus(n):
    ngpus = len(numba.cuda.gpus)
    return ngpus >= n


@pytest.mark.parametrize("delayed", [True, False])
def test_basic(loop, delayed):  # noqa: F811
    with dask_cuda.LocalCUDACluster(loop=loop) as cluster:
        with Client(cluster):
            pdf = dask.datasets.timeseries(dtypes={"x": int}).reset_index()
            gdf = pdf.map_partitions(cudf.DataFrame.from_pandas)
            if delayed:
                gdf = dd.from_delayed(gdf.to_delayed())
            assert_eq(pdf.head(), gdf.head())


def test_merge():
    # Repro Issue#3366
    with dask_cuda.LocalCUDACluster(n_workers=1) as cluster:
        with Client(cluster):
            r1 = cudf.DataFrame()
            r1["a1"] = range(4)
            r1["a2"] = range(4, 8)
            r1["a3"] = range(4)

            r2 = cudf.DataFrame()
            r2["b0"] = range(4)
            r2["b1"] = range(4)
            r2["b1"] = r2.b1.astype("str")

            d1 = dask_cudf.from_cudf(r1, 2)
            d2 = dask_cudf.from_cudf(r2, 2)

            res = d1.merge(d2, left_on=["a3"], right_on=["b0"])
            assert len(res) == 4


@pytest.mark.skipif(
    not at_least_n_gpus(2), reason="Machine does not have two GPUs"
)
def test_ucx_seriesgroupby():
    pytest.importorskip("ucp")

    # Repro Issue#3913
    with dask_cuda.LocalCUDACluster(n_workers=2, protocol="ucx") as cluster:
        with Client(cluster):
            df = cudf.DataFrame({"a": [1, 2, 3, 4], "b": [5, 1, 2, 5]})
            dask_df = dask_cudf.from_cudf(df, npartitions=2)
            dask_df_g = dask_df.groupby(["a"]).b.sum().compute()

            assert dask_df_g.name == "b"


def test_str_series_roundtrip():
    with dask_cuda.LocalCUDACluster(n_workers=1) as cluster:
        with Client(cluster):
            expected = cudf.Series(["hi", "hello", None])
            dask_series = dask_cudf.from_cudf(expected, npartitions=2)

            actual = dask_series.compute()
            assert_eq(actual, expected)


def test_p2p_shuffle():
    pytest.importorskip(
        "pyarrow",
        minversion="14.0.1",
        reason="P2P shuffling requires pyarrow>=14.0.1",
    )
    # Check that we can use `shuffle_method="p2p"`
    with dask_cuda.LocalCUDACluster(n_workers=1) as cluster:
        with Client(cluster):
            ddf = (
                dask.datasets.timeseries(
                    start="2000-01-01",
                    end="2000-01-08",
                    dtypes={"x": int},
                )
                .reset_index(drop=True)
                .to_backend("cudf")
            )
            dd.assert_eq(
                ddf.sort_values("x", shuffle_method="p2p").compute(),
                ddf.compute().sort_values("x"),
                check_index=False,
            )


@pytest.mark.skipif(
    not at_least_n_gpus(3),
    reason="Machine does not have three GPUs",
)
def test_unique():
    # Using `"p2p"` can produce dispatching problems
    # TODO: Test "p2p" after dask > 2024.4.1 is required
    # See: https://github.com/dask/dask/pull/11040
    with dask_cuda.LocalCUDACluster(n_workers=3) as cluster:
        with Client(cluster):
            df = cudf.DataFrame({"x": ["a", "b", "c", "a", "a"]})
            ddf = dask_cudf.from_cudf(df, npartitions=2)
            dd.assert_eq(
                df.x.unique(),
                ddf.x.unique().compute(),
                check_index=False,
            )


def test_serialization_of_numpy_types():
    # Dask uses numpy integers as column names, which can break cudf serialization
    with dask_cuda.LocalCUDACluster(n_workers=1) as cluster:
        with Client(cluster):
            with dask.config.set(
                {"dataframe.backend": "cudf", "array.backend": "cupy"}
            ):
                rng = da.random.default_rng()
                X_arr = rng.random((100, 10), chunks=(50, 10))
                X = dd.from_dask_array(X_arr)
                X = X[X.columns[0]]
                X.compute()

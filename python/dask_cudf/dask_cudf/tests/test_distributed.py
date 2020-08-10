import numba.cuda
import pytest

import dask
from dask import dataframe as dd
from dask.distributed import Client
from distributed.utils_test import loop  # noqa: F401

import dask_cudf

import cudf
from cudf.tests.utils import assert_eq

dask_cuda = pytest.importorskip("dask_cuda")


def more_than_two_gpus():
    ngpus = len(numba.cuda.gpus)
    return ngpus >= 2


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
    not more_than_two_gpus(), reason="Machine does not have more than two GPUs"
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

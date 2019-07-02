import pytest

import dask
import dask.dataframe as dd
from dask.distributed import Client
from distributed.utils_test import loop  # noqa: F401

import cudf

dask_cuda = pytest.importorskip("dask_cuda")


@pytest.mark.parametrize("delayed", [True, False])
def test_basic(loop, delayed):  # noqa: F811
    with dask_cuda.LocalCUDACluster(loop=loop) as cluster:
        with Client(cluster):
            pdf = dask.datasets.timeseries(dtypes={"x": int}).reset_index()
            gdf = pdf.map_partitions(cudf.DataFrame.from_pandas)
            if delayed:
                gdf = dd.from_delayed(gdf.to_delayed())
            dd.assert_eq(pdf.head(), gdf.head())

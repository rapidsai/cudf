# Copyright (c) 2018, NVIDIA CORPORATION.
import os.path

import numpy as np
import pyarrow as pa
import pytest
from numba import cuda

from cudf.comm.gpuarrow import GpuArrowReader
from cudf.core import DataFrame, Series
from cudf.tests.utils import assert_eq


def read_data():
    import pandas as pd

    basedir = os.path.dirname(__file__)
    datapath = os.path.join(basedir, "data", "ipums.pkl")
    try:
        df = pd.read_pickle(datapath)
    except Exception as excpr:
        if type(excpr).__name__ == "FileNotFoundError":
            pytest.skip(".pkl file is not found")
        else:
            print(type(excpr).__name__)

    names = []
    arrays = []
    for k in df.columns:
        arrays.append(pa.Array.from_pandas(df[k]))
        names.append(k)
    batch = pa.RecordBatch.from_arrays(arrays, names)
    schema = batch.schema.serialize().to_pybytes()
    schema = np.ndarray(
        shape=len(schema), dtype=np.byte, buffer=bytearray(schema)
    )
    data = batch.serialize().to_pybytes()
    data = np.ndarray(shape=len(data), dtype=np.byte, buffer=bytearray(data))
    darr = cuda.to_device(data)
    return df, schema, darr


def test_fillna():
    _, schema, darr = read_data()
    gar = GpuArrowReader(schema, darr)
    masked_col = gar[8]
    sr = Series(data=masked_col.data)
    dense = sr.nans_to_nulls().fillna(123)
    np.testing.assert_equal(123, dense.to_array())
    assert len(dense) == len(sr)
    assert dense.null_count == 0


def test_to_dense_array():
    data = np.random.random(8)
    mask = np.asarray([0b11010110], dtype=np.byte)

    sr = Series.from_masked_array(data=data, mask=mask, null_count=3)
    assert sr.has_nulls
    assert sr.null_count != len(sr)
    filled = sr.to_array(fillna="pandas")
    dense = sr.to_array()
    assert dense.size < filled.size
    assert filled.size == len(sr)


def test_reading_arrow_sparse_data():
    pdf, schema, darr = read_data()
    gar = GpuArrowReader(schema, darr)
    gdf = DataFrame(gar.to_dict())
    assert_eq(pdf, gdf)

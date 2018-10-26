# Copyright (c) 2018, NVIDIA CORPORATION.
import pytest
import os.path

try:
    import pyarrow as pa
    arrow_version = pa.__version__
except ImportError as msg:
    print('Failed to import pyarrow: {}'.format(msg))
    pa = None
    arrow_version = None

import numpy as np

from librmm_cffi import librmm as rmm

from cudf.gpuarrow import GpuArrowReader
from cudf.dataframe import Series, DataFrame


def read_data():
    import pandas as pd
    basedir = os.path.dirname(__file__)
    datapath = os.path.join(basedir, 'data', 'ipums.pkl')
    df = pd.read_pickle(datapath)
    names = []
    arrays = []
    for k in df.columns:
        arrays.append(pa.Array.from_pandas(df[k]))
        names.append(k)
    batch = pa.RecordBatch.from_arrays(arrays, names)
    schema = batch.schema.serialize().to_pybytes()
    schema = np.ndarray(shape=len(schema), dtype=np.byte,
                        buffer=bytearray(schema))
    data = batch.serialize().to_pybytes()
    data = np.ndarray(shape=len(data), dtype=np.byte,
                      buffer=bytearray(data))
    darr = rmm.to_device(data)
    return schema, darr


@pytest.mark.skipif(arrow_version is None,
                    reason='need compatible pyarrow to generate test data')
def test_fillna():
    schema, darr = read_data()
    gar = GpuArrowReader(schema, darr)
    masked_col = gar[8]
    assert masked_col.null_count
    sr = Series.from_masked_array(data=masked_col.data, mask=masked_col.null,
                                  null_count=masked_col.null_count)
    dense = sr.fillna(123)
    np.testing.assert_equal(123, dense.to_array())
    assert len(dense) == len(sr)
    assert dense.null_count == 0


def test_to_dense_array():
    data = np.random.random(8)
    mask = np.asarray([0b11010110], dtype=np.byte)

    sr = Series.from_masked_array(data=data, mask=mask, null_count=3)
    assert sr.null_count > 0
    assert sr.null_count != len(sr)
    filled = sr.to_array(fillna='pandas')
    dense = sr.to_array()
    assert dense.size < filled.size
    assert filled.size == len(sr)


@pytest.mark.skipif(arrow_version is None,
                    reason='need compatible pyarrow to generate test data')
def test_reading_arrow_sparse_data():
    schema, darr = read_data()
    gar = GpuArrowReader(schema, darr)

    df = DataFrame(gar.to_dict().items())

    # preprocessing
    num_cols = set()
    cat_cols = set()
    response_set = set(['INCEARN '])
    feature_names = set(df.columns) - response_set

    # Determine cat and numeric columns
    uniques = {}
    for k in feature_names:
        try:
            uniquevals = df[k].unique()
            uniques[k] = uniquevals
        except ValueError:
            num_cols.add(k)
        else:
            nunique = len(uniquevals)
            if nunique < 2:
                del df[k]
            elif 1 < nunique < 1000:
                cat_cols.add(k)
            else:
                num_cols.add(k)

    # Fix numeric columns
    for k in (num_cols - response_set):
        df[k] = df[k].fillna(df[k].mean())
        assert df[k].null_count == 0
        std = df[k].std()
        # drop near constant columns
        if not np.isfinite(std) or std < 1e-4:
            del df[k]
            print('drop near constant', k)
        else:
            df[k] = df[k].scale()

    # Expand categorical columns
    for k in cat_cols:
        cats = uniques[k][1:]  # drop first
        df = df.one_hot_encoding(k, prefix=k, cats=cats)
        del df[k]

    # Print dtypes
    assert {df[k].dtype for k in df.columns} == {np.dtype('float64')}

    mat = df.as_matrix()

    assert mat.max() == 1
    assert mat.min() == 0


if __name__ == '__main__':
    test_reading_arrow_sparse_data()

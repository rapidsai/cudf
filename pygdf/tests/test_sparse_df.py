import os.path
import pickle

import numpy as np
from numba import cuda

from pygdf.gpuarrow import GpuArrowReader
from pygdf.dataframe import Series, DataFrame


def read_data():
    basedir = os.path.dirname(__file__)
    # load schema
    schemapath = os.path.join(basedir, 'data', 'schema_ipums.pickle')
    with open(schemapath, 'rb') as fin:
        schema = pickle.load(fin)
    # load data
    datapath = os.path.join(basedir, 'data', 'data_ipums.pickle')
    with open(datapath, 'rb') as fin:
        data = pickle.load(fin)
    darr = cuda.to_device(data)
    return schema, darr


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
    assert not dense.has_null_mask


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

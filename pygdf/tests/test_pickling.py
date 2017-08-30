import sys
import pickle

import numpy as np
from numba import cuda
import pandas as pd

from pygdf.dataframe import DataFrame, GenericIndex


def check_serialization(df):
    # basic
    assert_frame_picklable(df)
    # sliced
    assert_frame_picklable(df[:-1])
    assert_frame_picklable(df[1:])
    assert_frame_picklable(df[2:-2])
    # sorted
    sortvaldf = df.sort_values('vals')
    assert isinstance(sortvaldf.index, GenericIndex)
    assert_frame_picklable(sortvaldf)


def assert_frame_picklable(df):
    serialbytes = pickle.dumps(df)
    loaded = pickle.loads(serialbytes)
    pd.util.testing.assert_frame_equal(loaded.to_pandas(), df.to_pandas())


def test_pickle_dataframe_numeric():
    np.random.seed(0)
    df = DataFrame()
    nelem = 10
    df['keys'] = np.arange(nelem, dtype=np.float64)
    df['vals'] = np.random.random(nelem)

    check_serialization(df)


def test_pickle_dataframe_categorical():
    np.random.seed(0)

    df = DataFrame()
    df['keys'] = pd.Categorical("aaabababac")
    df['vals'] = np.random.random(len(df))

    check_serialization(df)


def test_sizeof_dataframe():
    np.random.seed(0)
    df = DataFrame()
    nelem = 1000
    df['keys'] = hkeys = np.arange(nelem, dtype=np.float64)
    df['vals'] = hvals = np.random.random(nelem)

    nbytes = hkeys.nbytes + hvals.nbytes
    sizeof = sys.getsizeof(df)
    assert sizeof >= nbytes

    serialized_nbytes = len(pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL))
    # Serialized size should be close to what __sizeof__ is giving
    np.testing.assert_approx_equal(sizeof, serialized_nbytes, significant=2)


def test_pickle_index():
    nelem = 10
    idx = GenericIndex(cuda.to_device(np.arange(nelem)))
    pickled = pickle.dumps(idx)
    out = pickle.loads(pickled)
    assert idx == out

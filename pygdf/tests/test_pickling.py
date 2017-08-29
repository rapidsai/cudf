import sys
import pickle

import numpy as np
import pandas as pd

from pygdf.dataframe import Series, DataFrame


def check_serialization(df):
    assert_frame_picklable(df)
    assert_frame_picklable(df[:-1])
    assert_frame_picklable(df[1:])
    assert_frame_picklable(df[2:-2])


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

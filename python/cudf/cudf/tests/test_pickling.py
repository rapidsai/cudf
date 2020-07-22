# Copyright (c) 2018, NVIDIA CORPORATION.

import sys

import numpy as np
import pandas as pd
import pytest

from cudf.core import DataFrame, GenericIndex, Series
from cudf.core.buffer import Buffer
from cudf.tests.utils import assert_eq

if sys.version_info < (3, 8):
    try:
        import pickle5 as pickle
    except ImportError:
        import pickle
else:
    import pickle


def check_serialization(df):
    # basic
    assert_frame_picklable(df)
    # sliced
    assert_frame_picklable(df[:-1])
    assert_frame_picklable(df[1:])
    assert_frame_picklable(df[2:-2])
    # sorted
    sortvaldf = df.sort_values("vals")
    assert isinstance(sortvaldf.index, GenericIndex)
    assert_frame_picklable(sortvaldf)
    # out-of-band
    if pickle.HIGHEST_PROTOCOL >= 5:
        buffers = []
        serialbytes = pickle.dumps(
            df, protocol=5, buffer_callback=buffers.append
        )
        for b in buffers:
            assert isinstance(b, pickle.PickleBuffer)
        loaded = pickle.loads(serialbytes, buffers=buffers)
        assert_eq(loaded, df)


def assert_frame_picklable(df):
    serialbytes = pickle.dumps(df)
    loaded = pickle.loads(serialbytes)
    assert_eq(loaded, df)


def test_pickle_dataframe_numeric():
    np.random.seed(0)
    df = DataFrame()
    nelem = 10
    df["keys"] = np.arange(nelem, dtype=np.float64)
    df["vals"] = np.random.random(nelem)

    check_serialization(df)


def test_pickle_dataframe_categorical():
    np.random.seed(0)

    df = DataFrame()
    df["keys"] = pd.Categorical("aaabababac")
    df["vals"] = np.random.random(len(df))

    check_serialization(df)


def test_sizeof_dataframe():
    np.random.seed(0)
    df = DataFrame()
    nelem = 1000
    df["keys"] = hkeys = np.arange(nelem, dtype=np.float64)
    df["vals"] = hvals = np.random.random(nelem)

    nbytes = hkeys.nbytes + hvals.nbytes
    sizeof = sys.getsizeof(df)
    assert sizeof >= nbytes

    serialized_nbytes = len(pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL))

    # assert at least sizeof bytes were serialized
    assert serialized_nbytes >= sizeof


def test_pickle_index():
    nelem = 10
    idx = GenericIndex(np.arange(nelem), name="a")
    pickled = pickle.dumps(idx)
    out = pickle.loads(pickled)
    assert idx == out


def test_pickle_buffer():
    arr = np.arange(10).view("|u1")
    buf = Buffer(arr)
    assert buf.size == arr.nbytes
    pickled = pickle.dumps(buf)
    unpacked = pickle.loads(pickled)
    # Check that unpacked capacity equals buf.size
    assert unpacked.size == arr.nbytes


@pytest.mark.parametrize("named", [True, False])
def test_pickle_series(named):
    np.random.seed(0)
    if named:
        ser = Series(np.random.random(10), name="a")
    else:
        ser = Series(np.random.random(10))

    pickled = pickle.dumps(ser)
    out = pickle.loads(pickled)
    assert (ser == out).all()


@pytest.mark.parametrize(
    "slices",
    [
        slice(None, None, None),
        slice(1, 3, 1),
        slice(0, 3, 1),
        slice(3, 5, 1),
        slice(10, 12, 1),
    ],
)
def test_pickle_categorical_column(slices):
    sr = Series(["a", "b", None, "a", "c", "b"]).astype("category")
    sliced_sr = sr.iloc[slices]
    input_col = sliced_sr._column

    pickled = pickle.dumps(input_col)
    out = pickle.loads(pickled)

    assert_eq(Series(out), Series(input_col))


@pytest.mark.parametrize(
    "slices",
    [
        slice(None, None, None),
        slice(1, 3, 1),
        slice(0, 3, 1),
        slice(3, 5, 1),
        slice(10, 12, 1),
    ],
)
def test_pickle_string_column(slices):
    sr = Series(["a", "b", None, "a", "c", "b"])
    sliced_sr = sr.iloc[slices]
    input_col = sliced_sr._column

    pickled = pickle.dumps(input_col)
    out = pickle.loads(pickled)

    assert_eq(Series(out), Series(input_col))

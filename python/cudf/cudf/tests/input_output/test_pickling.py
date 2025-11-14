# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import io
import pickle

import numpy as np
import pandas as pd
import pytest

from cudf import DataFrame, Index, RangeIndex, Series
from cudf.core.buffer import as_buffer
from cudf.testing import assert_eq

pytestmark = pytest.mark.spilling


@pytest.mark.parametrize(
    "keys",
    [
        np.arange(5, dtype=np.float64),
        pd.Categorical(["a", "a", "a", "b", "a", "b", "a", "b", "a", "c"]),
    ],
)
def test_pickle_dataframe(keys):
    rng = np.random.default_rng(seed=0)
    df = DataFrame({"keys": keys, "vals": rng.random(len(keys))})
    # basic
    assert_frame_picklable(df)
    # sliced
    assert_frame_picklable(df[:-1])
    assert_frame_picklable(df[1:])
    assert_frame_picklable(df[2:-2])
    # sorted
    sortvaldf = df.sort_values("vals")
    assert isinstance(sortvaldf.index, (Index, RangeIndex))
    assert_frame_picklable(sortvaldf)
    # out-of-band
    buffers = []
    serialbytes = pickle.dumps(df, protocol=5, buffer_callback=buffers.append)
    for b in buffers:
        assert isinstance(b, pickle.PickleBuffer)
    loaded = pickle.loads(serialbytes, buffers=buffers)
    assert_eq(loaded, df)


def assert_frame_picklable(df):
    serialbytes = pickle.dumps(df)
    loaded = pickle.loads(serialbytes)
    assert_eq(loaded, df)


def test_memory_usage_dataframe():
    rng = np.random.default_rng(seed=0)
    df = DataFrame()
    nelem = 1000
    df["keys"] = hkeys = np.arange(nelem, dtype=np.float64)
    df["vals"] = hvals = rng.random(nelem)

    nbytes = hkeys.nbytes + hvals.nbytes
    sizeof = df.memory_usage().sum()
    assert sizeof >= nbytes

    serialized_nbytes = len(pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL))

    # assert at least sizeof bytes were serialized
    assert serialized_nbytes >= sizeof


def test_pickle_index():
    nelem = 10
    idx = Index(np.arange(nelem), name="a")
    pickled = pickle.dumps(idx)
    out = pickle.loads(pickled)
    assert_eq(idx, out)


def test_pickle_buffer():
    arr = np.arange(10).view("|u1")
    buf = as_buffer(arr)
    assert buf.size == arr.nbytes
    pickled = pickle.dumps(buf)
    unpacked = pickle.loads(pickled)
    # Check that unpacked capacity equals buf.size
    assert unpacked.size == arr.nbytes


@pytest.mark.parametrize("name", [None, "a"])
def test_pickle_series(name):
    rng = np.random.default_rng(seed=0)
    ser = Series(rng.random(10), name=name)
    pickled = pickle.dumps(ser)
    out = pickle.loads(pickled)
    assert_eq(ser, out)


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

    assert_eq(Series._from_column(out), Series._from_column(input_col))


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

    assert_eq(Series._from_column(out), Series._from_column(input_col))


@pytest.mark.parametrize(
    "names",
    [
        ["a", "b", "c"],
        [None, None, None],
        ["aa", "aa", "aa"],
        ["bb", "aa", "aa"],
        None,
    ],
)
def test_pickle_roundtrip_multiindex(names):
    df = DataFrame(
        {
            "one": [1, 2, 3],
            "two": [True, False, True],
            "three": ["ab", "cd", "ef"],
            "four": [0.2, 0.1, -10.2],
        }
    )
    expected_df = df.set_index(["one", "two", "three"])
    expected_df.index.names = names
    local_file = io.BytesIO()

    pickle.dump(expected_df, local_file)
    local_file.seek(0)
    actual_df = pickle.load(local_file)
    assert_eq(expected_df, actual_df)

# Copyright (c) 2021-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import sys

import numpy as np
import pandas as pd

from cudf import DataFrame, Index, Series
from cudf._lib.copying import pack, unpack
from cudf.testing import assert_eq


def test_sizeof_packed_dataframe():
    rng = np.random.default_rng(seed=0)
    df = DataFrame()
    nelem = 1000
    df["keys"] = hkeys = np.arange(nelem, dtype=np.float64)
    df["vals"] = hvals = rng.random(nelem)
    packed = pack(df)

    nbytes = hkeys.nbytes + hvals.nbytes
    sizeof = sys.getsizeof(packed)
    assert sizeof < nbytes

    serialized_nbytes = len(
        pickle.dumps(packed, protocol=pickle.HIGHEST_PROTOCOL)
    )

    # assert at least sizeof bytes were serialized
    assert serialized_nbytes >= sizeof


def check_packed_equality(df):
    # basic
    assert_packed_frame_equality(df)
    # sliced
    assert_packed_frame_equality(df[:-1])
    assert_packed_frame_equality(df[1:])
    assert_packed_frame_equality(df[2:-2])
    # sorted
    sortvaldf = df.sort_values("vals")
    assert isinstance(sortvaldf.index, Index)
    assert_packed_frame_equality(sortvaldf)


def assert_packed_frame_equality(df):
    pdf = df.to_pandas()

    packed = pack(df)
    del df
    unpacked = unpack(packed)

    assert_eq(unpacked, pdf)


def test_packed_dataframe_equality_numeric():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    nelem = 10
    df["keys"] = np.arange(nelem, dtype=np.float64)
    df["vals"] = rng.random(nelem)

    check_packed_equality(df)


def test_packed_dataframe_equality_categorical():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = pd.Categorical(
        ["a", "a", "a", "b", "a", "b", "a", "b", "a", "c"]
    )
    df["vals"] = rng.random(len(df))

    check_packed_equality(df)


def test_packed_dataframe_equality_list():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = Series(list([i, i + 1, i + 2] for i in range(10)))
    df["vals"] = rng.random(len(df))

    check_packed_equality(df)


def test_packed_dataframe_equality_struct():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = Series(
        list({"0": i, "1": i + 1, "2": i + 2} for i in range(10))
    )
    df["vals"] = rng.random(len(df))

    check_packed_equality(df)


def check_packed_unique_pointers(df):
    # basic
    assert_packed_frame_unique_pointers(df)
    # sliced
    assert_packed_frame_unique_pointers(df[:-1])
    assert_packed_frame_unique_pointers(df[1:])
    assert_packed_frame_unique_pointers(df[2:-2])
    # sorted
    sortvaldf = df.sort_values("vals")
    assert isinstance(sortvaldf.index, Index)
    assert_packed_frame_unique_pointers(sortvaldf)


def assert_packed_frame_unique_pointers(df):
    unpacked = unpack(pack(df))

    for col in df:
        if df._data[col].data:
            assert df._data[col].data.get_ptr(mode="read") != unpacked._data[
                col
            ].data.get_ptr(mode="read")


def test_packed_dataframe_unique_pointers_numeric():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    nelem = 10
    df["keys"] = np.arange(nelem, dtype=np.float64)
    df["vals"] = rng.random(nelem)

    check_packed_unique_pointers(df)


def test_packed_dataframe_unique_pointers_categorical():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = pd.Categorical(
        ["a", "a", "a", "b", "a", "b", "a", "b", "a", "c"]
    )
    df["vals"] = rng.random(len(df))

    check_packed_unique_pointers(df)


def test_packed_dataframe_unique_pointers_list():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = Series(list([i, i + 1, i + 2] for i in range(10)))
    df["vals"] = rng.random(len(df))

    check_packed_unique_pointers(df)


def test_packed_dataframe_unique_pointers_struct():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = Series(
        list({"0": i, "1": i + 1, "2": i + 2} for i in range(10))
    )
    df["vals"] = rng.random(len(df))

    check_packed_unique_pointers(df)


def check_packed_pickled_equality(df):
    # basic
    assert_packed_frame_picklable(df)
    # sliced
    assert_packed_frame_picklable(df[:-1])
    assert_packed_frame_picklable(df[1:])
    assert_packed_frame_picklable(df[2:-2])
    # sorted
    sortvaldf = df.sort_values("vals")
    assert isinstance(sortvaldf.index, Index)
    assert_packed_frame_picklable(sortvaldf)
    # out-of-band
    buffers = []
    serialbytes = pickle.dumps(
        pack(df), protocol=5, buffer_callback=buffers.append
    )
    for b in buffers:
        assert isinstance(b, pickle.PickleBuffer)
    loaded = unpack(pickle.loads(serialbytes, buffers=buffers))
    assert_eq(loaded, df)


def assert_packed_frame_picklable(df):
    serialbytes = pickle.dumps(pack(df))
    loaded = unpack(pickle.loads(serialbytes))
    assert_eq(loaded, df)


def test_pickle_packed_dataframe_numeric():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    nelem = 10
    df["keys"] = np.arange(nelem, dtype=np.float64)
    df["vals"] = rng.random(nelem)

    check_packed_pickled_equality(df)


def test_pickle_packed_dataframe_categorical():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = pd.Categorical(
        ["a", "a", "a", "b", "a", "b", "a", "b", "a", "c"]
    )
    df["vals"] = rng.random(len(df))

    check_packed_pickled_equality(df)


def test_pickle_packed_dataframe_list():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = Series(list([i, i + 1, i + 2] for i in range(10)))
    df["vals"] = rng.random(len(df))

    check_packed_pickled_equality(df)


def test_pickle_packed_dataframe_struct():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = Series(
        list({"0": i, "1": i + 1, "2": i + 2} for i in range(10))
    )
    df["vals"] = rng.random(len(df))

    check_packed_pickled_equality(df)


def check_packed_serialized_equality(df):
    # basic
    assert_packed_frame_serializable(df)
    # sliced
    assert_packed_frame_serializable(df[:-1])
    assert_packed_frame_serializable(df[1:])
    assert_packed_frame_serializable(df[2:-2])
    # sorted
    sortvaldf = df.sort_values("vals")
    assert isinstance(sortvaldf.index, Index)
    assert_packed_frame_serializable(sortvaldf)


def assert_packed_frame_serializable(df):
    packed = pack(df)
    header, frames = packed.serialize()
    loaded = unpack(packed.deserialize(header, frames))
    assert_eq(loaded, df)


def test_serialize_packed_dataframe_numeric():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    nelem = 10
    df["keys"] = np.arange(nelem, dtype=np.float64)
    df["vals"] = rng.random(nelem)

    check_packed_serialized_equality(df)


def test_serialize_packed_dataframe_categorical():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = pd.Categorical(
        ["a", "a", "a", "b", "a", "b", "a", "b", "a", "c"]
    )
    df["vals"] = rng.random(len(df))

    check_packed_serialized_equality(df)


def test_serialize_packed_dataframe_list():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = Series(list([i, i + 1, i + 2] for i in range(10)))
    df["vals"] = rng.random(len(df))

    check_packed_serialized_equality(df)


def test_serialize_packed_dataframe_struct():
    rng = np.random.default_rng(seed=0)

    df = DataFrame()
    df["keys"] = Series(
        list({"0": i, "1": i + 1, "2": i + 2} for i in range(10))
    )
    df["vals"] = rng.random(len(df))

    check_packed_serialized_equality(df)

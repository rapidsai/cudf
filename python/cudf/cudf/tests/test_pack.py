# Copyright (c) 2021, NVIDIA CORPORATION.
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

import numpy as np
import pandas as pd

from cudf._lib.copying import pack, unpack
from cudf.core import DataFrame, GenericIndex, Series
from cudf.tests.utils import assert_eq


def check_packed_equality(df):
    # basic
    assert_packed_frame_equality(df)
    # sliced
    assert_packed_frame_equality(df[:-1])
    assert_packed_frame_equality(df[1:])
    assert_packed_frame_equality(df[2:-2])
    # sorted
    sortvaldf = df.sort_values("vals")
    assert isinstance(sortvaldf.index, GenericIndex)
    assert_packed_frame_equality(sortvaldf)


def assert_packed_frame_equality(df):
    pdf = df.to_pandas()

    packed = pack(df)
    del df
    unpacked = unpack(packed)

    assert_eq(unpacked, pdf)


def test_packed_dataframe_equality_numeric():
    np.random.seed(0)
    df = DataFrame()
    nelem = 10
    df["keys"] = np.arange(nelem, dtype=np.float64)
    df["vals"] = np.random.random(nelem)

    check_packed_equality(df)


def test_packed_dataframe_equality_categorical():
    np.random.seed(0)

    df = DataFrame()
    df["keys"] = pd.Categorical("aaabababac")
    df["vals"] = np.random.random(len(df))

    check_packed_equality(df)


def test_packed_dataframe_equality_list():
    np.random.seed(0)

    df = DataFrame()
    df["keys"] = Series(list([i, i + 1, i + 2] for i in range(10)))
    df["vals"] = np.random.random(len(df))

    check_packed_equality(df)


def test_packed_dataframe_equality_struct():
    np.random.seed(0)

    df = DataFrame()
    df["keys"] = Series(
        list({"0": i, "1": i + 1, "2": i + 2} for i in range(10))
    )
    df["vals"] = np.random.random(len(df))

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
    assert isinstance(sortvaldf.index, GenericIndex)
    assert_packed_frame_unique_pointers(sortvaldf)


def assert_packed_frame_unique_pointers(df):
    unpacked = unpack(pack(df))

    for col in df:
        if df._data[col].data:
            assert df._data[col].data.ptr != unpacked._data[col].data.ptr


def test_packed_dataframe_unique_pointers_numeric():
    np.random.seed(0)
    df = DataFrame()
    nelem = 10
    df["keys"] = np.arange(nelem, dtype=np.float64)
    df["vals"] = np.random.random(nelem)

    check_packed_unique_pointers(df)


def test_packed_dataframe_unique_pointers_categorical():
    np.random.seed(0)

    df = DataFrame()
    df["keys"] = pd.Categorical("aaabababac")
    df["vals"] = np.random.random(len(df))

    check_packed_unique_pointers(df)

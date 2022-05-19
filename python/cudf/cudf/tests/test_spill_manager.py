# Copyright (c) 2022, NVIDIA CORPORATION.


import rmm

import cudf
from cudf.core.buffer import Buffer


def test_spillable_buffer():
    buf = Buffer(rmm.DeviceBuffer(size=10), sole_owner=True)
    assert buf.spillable
    buf.ptr  # Expose pointer
    assert buf._raw_pointer_exposed
    assert not buf.spillable
    buf = Buffer(rmm.DeviceBuffer(size=10), sole_owner=True)
    buf.__cuda_array_interface__  # Expose pointer
    assert buf._raw_pointer_exposed
    assert not buf.spillable


def test_spillable_df_creation():
    df = cudf.datasets.timeseries()
    assert df._data._data["x"].data.spillable
    df = cudf.DataFrame({"x": [1, 2, 3]})
    assert df._data._data["x"].data.spillable
    df = cudf.datasets.randomdata(10)
    assert df._data._data["x"].data.spillable


def test_spillable_df_groupby():
    df = cudf.DataFrame({"x": [1, 1, 1]})
    gb = df.groupby("x")
    # `gd` holds a reference to the device memory, which makes
    # the buffer unspillable
    assert df._data._data["x"].data._access_counter.use_count() == 2
    assert not df._data._data["x"].data.spillable
    del gb
    assert df._data._data["x"].data.spillable

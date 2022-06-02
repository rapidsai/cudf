# Copyright (c) 2022, NVIDIA CORPORATION.


import warnings

import pytest

import rmm

import cudf
from cudf.core.buffer import Buffer
from cudf.core.spill_manager import SpillManager, global_manager


def gen_df() -> cudf.DataFrame:
    return cudf.DataFrame({"a": [1, 2, 3]})


gen_df.buffer = lambda df: df._data._data["a"].data
gen_df.is_spilled = lambda df: gen_df.buffer(df).is_spilled
gen_df.is_spillable = lambda df: gen_df.buffer(df).spillable
gen_df.buffer_size = gen_df.buffer(gen_df()).size


@pytest.fixture
def manager(request):
    kwargs = dict(getattr(request, "param", {}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            yield global_manager.reset(SpillManager(**kwargs))
        finally:
            global_manager.clear()


def test_spillable_buffer():
    buf = Buffer(rmm.DeviceBuffer(size=10), sole_owner=True)
    assert buf.spillable
    buf.ptr  # Expose pointer
    assert buf.ptr_exposed
    assert not buf.spillable
    buf = Buffer(rmm.DeviceBuffer(size=10), sole_owner=True)
    buf.__cuda_array_interface__  # Expose pointer
    assert buf.ptr_exposed
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
    # `gb` holds a reference to the device memory, which makes
    # the buffer unspillable
    assert df._data._data["x"].data._access_counter.use_count() == 2
    assert not df._data._data["x"].data.spillable
    del gb
    assert df._data._data["x"].data.spillable


def test_spillable_df_views():
    df = gen_df()
    assert gen_df.is_spillable(df)
    df_view = df.loc[1:]
    # TODO: support spillable views, for now we mark the view
    #       and its base as unspillable
    assert not gen_df.is_spillable(df_view)
    assert not gen_df.is_spillable(df)


def test_spilling_buffer():
    buf = Buffer(rmm.DeviceBuffer(size=10), sole_owner=True)
    buf.move_inplace(target="cpu")
    assert buf.is_spilled
    buf.ptr  # Expose pointer and trigger unspill
    assert not buf.is_spilled
    with pytest.raises(ValueError, match="unspillable buffer"):
        buf.move_inplace(target="cpu")


def test_environment_variables(monkeypatch):
    monkeypatch.setenv("CUDF_SPILL_ON_DEMAND", "off")
    monkeypatch.setenv("CUDF_SPILL", "off")
    with pytest.raises(ValueError, match="No global SpillManager"):
        global_manager.get()
    assert not global_manager.enabled
    monkeypatch.setenv("CUDF_SPILL", "on")
    assert not global_manager.enabled
    global_manager.clear()  # Trigger re-read of environment variables
    assert global_manager.enabled
    manager = global_manager.get()
    assert manager._spill_on_demand is False
    assert manager._device_memory_limit is None
    global_manager.clear()
    monkeypatch.setenv("CUDF_SPILL_DEVICE_LIMIT", "1000")
    manager = global_manager.get()
    assert manager._spill_on_demand is False
    assert isinstance(manager._device_memory_limit, int)
    assert manager._device_memory_limit == 1000


def test_spill_device_memory(manager: SpillManager):
    df = gen_df()
    assert manager.spilled_and_unspilled() == (0, gen_df.buffer_size)
    manager.spill_device_memory()
    assert manager.spilled_and_unspilled() == (gen_df.buffer_size, 0)
    del df
    assert manager.spilled_and_unspilled() == (0, 0)
    df1 = gen_df()
    df2 = gen_df()
    manager.spill_device_memory()
    assert gen_df.is_spilled(df1)
    assert not gen_df.is_spilled(df2)
    manager.spill_device_memory()
    assert gen_df.is_spilled(df1)
    assert gen_df.is_spilled(df2)
    df3 = df1 + df2
    assert not gen_df.is_spilled(df1)
    assert not gen_df.is_spilled(df2)
    assert not gen_df.is_spilled(df3)
    manager.spill_device_memory()
    assert gen_df.is_spilled(df1)
    assert not gen_df.is_spilled(df2)
    assert not gen_df.is_spilled(df3)
    df2.abs()  # Should change the access time
    manager.spill_device_memory()
    assert gen_df.is_spilled(df1)
    assert not gen_df.is_spilled(df2)
    assert gen_df.is_spilled(df3)


def test_spill_to_device_limit(manager: SpillManager):
    df1 = gen_df()
    df2 = gen_df()
    assert manager.spilled_and_unspilled() == (0, gen_df.buffer_size * 2)
    manager.spill_to_device_limit(device_limit=0)
    assert manager.spilled_and_unspilled() == (gen_df.buffer_size * 2, 0)
    df3 = df1 + df2
    manager.spill_to_device_limit(device_limit=0)
    assert manager.spilled_and_unspilled() == (gen_df.buffer_size * 3, 0)
    assert gen_df.is_spilled(df1)
    assert gen_df.is_spilled(df2)
    assert gen_df.is_spilled(df3)


@pytest.mark.parametrize(
    "manager", [{"device_memory_limit": 0}], indirect=True
)
def test_zero_device_limit(manager: SpillManager):
    assert manager._device_memory_limit == 0
    df1 = gen_df()
    df2 = gen_df()
    assert manager.spilled_and_unspilled() == (gen_df.buffer_size * 2, 0)
    df1 + df2
    # Notice, while performing the addintion both df1 and df2 are unspillable
    assert manager.spilled_and_unspilled() == (0, gen_df.buffer_size * 2)
    manager.spill_to_device_limit()
    assert manager.spilled_and_unspilled() == (gen_df.buffer_size * 2, 0)


def test_lookup_address_range(manager: SpillManager):
    df = gen_df()
    buffers = manager.base_buffers()
    assert len(buffers) == 1
    (buf,) = buffers
    assert gen_df.buffer(df) is buf
    assert manager.lookup_address_range(buf.ptr, buf.size) is buf
    assert manager.lookup_address_range(buf.ptr + 1, buf.size - 1) is buf
    assert manager.lookup_address_range(buf.ptr + 1, buf.size + 1) is buf
    assert manager.lookup_address_range(buf.ptr - 1, buf.size - 1) is buf
    assert manager.lookup_address_range(buf.ptr - 1, buf.size + 1) is buf
    assert manager.lookup_address_range(buf.ptr + buf.size, buf.size) is None
    assert manager.lookup_address_range(buf.ptr - buf.size, buf.size) is None


def test_spilling_df_views():
    df = gen_df()
    assert gen_df.is_spillable(df)
    gen_df.buffer(df).move_inplace(target="cpu")
    assert gen_df.is_spilled(df)
    df_view = df.loc[1:]
    assert not gen_df.is_spillable(df_view)
    assert not gen_df.is_spillable(df)

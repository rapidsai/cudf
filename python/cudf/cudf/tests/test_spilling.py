# Copyright (c) 2022, NVIDIA CORPORATION.


import gc
import warnings

import numpy as np
import pandas
import pandas.testing
import pytest

import rmm

import cudf
from cudf._lib.spillable_buffer import SpillableBuffer, SpillLock
from cudf.core.abc import Serializable
from cudf.core.buffer import Buffer, DeviceBufferLike, as_device_buffer_like
from cudf.core.spill_manager import (
    SpillManager,
    get_columns,
    global_manager,
    mark_columns_as_read_only_inplace,
)
from cudf.testing._utils import assert_eq


def gen_df(target="gpu") -> cudf.DataFrame:
    ret = cudf.DataFrame({"a": [1, 2, 3]})
    if target != "gpu":
        gen_df.buffer(ret).__spill__(target=target)
    return ret


gen_df.buffer = lambda df: df._data._data["a"].data
gen_df.is_spilled = lambda df: gen_df.buffer(df).is_spilled
gen_df.is_spillable = lambda df: gen_df.buffer(df).spillable
gen_df.buffer_size = gen_df.buffer(gen_df()).size


@pytest.fixture
def manager(request):
    """Fixture to enable and make a spilling manager availabe"""
    kwargs = dict(getattr(request, "param", {}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield global_manager.reset(SpillManager(**kwargs))
        if request.node.report["call"].failed:
            # Ignore `overwriting non-empty manager` errors when
            # test is failing.
            warnings.simplefilter("ignore")
        global_manager.clear()


def test_spillable_buffer(manager: SpillManager):
    buf = SpillableBuffer(
        data=rmm.DeviceBuffer(size=10), exposed=False, manager=manager
    )
    assert isinstance(buf, DeviceBufferLike)
    assert buf.spillable
    buf.ptr  # Expose pointer
    assert buf.exposed
    assert not buf.spillable
    buf = SpillableBuffer(
        data=rmm.DeviceBuffer(size=10), exposed=False, manager=manager
    )
    # Notice, accessing `__cuda_array_interface__` itself doesn't
    # expose the pointer, only accessing the "data" field exposes
    # the pointer.
    iface = buf.__cuda_array_interface__
    assert not buf.exposed
    assert buf.spillable
    iface["data"][0]  # Expose pointer
    assert buf.exposed
    assert not buf.spillable


def test_from_pandas(manager: SpillManager):
    pdf1 = pandas.DataFrame({"x": [1, 2, 3]})
    df = cudf.from_pandas(pdf1)
    assert df._data._data["x"].data.spillable
    pdf2 = df.to_pandas()
    pandas.testing.assert_frame_equal(pdf1, pdf2)


def test_creations(manager: SpillManager):
    df = cudf.datasets.timeseries()
    assert isinstance(df._data._data["x"].data, SpillableBuffer)
    assert df._data._data["x"].data.spillable
    df = cudf.DataFrame({"x": [1, 2, 3]})
    assert df._data._data["x"].data.spillable
    df = cudf.datasets.randomdata(10)
    assert df._data._data["x"].data.spillable


def test_spillable_df_groupby(manager: SpillManager):
    df = cudf.DataFrame({"x": [1, 1, 1]})
    gb = df.groupby("x")
    # `gb` holds a reference to the device memory, which makes
    # the buffer unspillable
    assert df._data._data["x"].data.expose_counter == 2
    assert not df._data._data["x"].data.spillable
    del gb
    assert df._data._data["x"].data.spillable


def test_spilling_buffer(manager: SpillManager):
    buf = as_device_buffer_like(rmm.DeviceBuffer(size=10), exposed=False)
    buf.__spill__(target="cpu")
    assert buf.is_spilled
    buf.ptr  # Expose pointer and trigger unspill
    assert not buf.is_spilled
    with pytest.raises(ValueError, match="unspillable buffer"):
        buf.__spill__(target="cpu")


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
    global_manager.clear()
    monkeypatch.setenv("CUDF_SPILL_STAT_EXPOSE", "on")
    manager = global_manager.get()
    assert isinstance(manager._expose_statistics, dict)


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
    buf = gen_df.buffer(df)
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


def test_external_memory_never_spills(manager):
    """
    Test that external data, i.e., data not managed by RMM,
    is never spilled
    """

    cp = pytest.importorskip("cupy")
    cp.cuda.set_allocator()  # uses default allocator

    a = cp.asarray([1, 2, 3])
    s = cudf.Series(a)
    assert len(manager.base_buffers()) == 0
    assert not s._data[None].data.spillable


def test_spilling_df_views(manager):
    df = gen_df()
    assert gen_df.is_spillable(df)
    gen_df.buffer(df).__spill__(target="cpu")
    assert gen_df.is_spilled(df)
    df_view = df.loc[1:]
    assert gen_df.is_spillable(df_view)
    assert gen_df.is_spillable(df)


def test_modify_spilled_views(manager):
    df = gen_df()
    df_view = df.iloc[1:]
    buf = gen_df.buffer(df)
    buf.__spill__(target="cpu")

    # modify the spilled df and check that the changes are reflected
    # in the view
    df.iloc[1:] = 0
    assert_eq(df_view, df.iloc[1:])

    # now, modify the view and check that the changes are reflected in
    # the df
    df_view.iloc[:] = -1
    assert_eq(df_view, df.iloc[1:])


def test_get_columns():
    df1 = cudf.DataFrame({"a": [1, 2, 3]})
    df2 = cudf.DataFrame({"b": [4, 5, 6], "c": [7, 8, 9]})
    cols = get_columns(({"x": [df1, df2], "y": [df2]},))
    assert len(cols) == 3
    assert cols[0] is df1._data["a"]
    assert cols[1] is df2._data["b"]
    assert cols[2] is df2._data["c"]


def test_mark_columns_as_read_only(manager: SpillManager):
    df_base = cudf.DataFrame({"a": range(10)})
    df_views = df_base.iloc[0:1], df_base.iloc[1:3]
    manager.spill_to_device_limit(0)
    assert len(manager.base_buffers()) == 1

    mark_columns_as_read_only_inplace(df_views)
    assert len(manager.base_buffers()) == 3
    del df_base
    gc.collect()
    assert len(manager.base_buffers()) == 2

    assert_eq(df_views[0], cudf.DataFrame({"a": range(10)}).iloc[0:1])
    assert_eq(df_views[1], cudf.DataFrame({"a": range(10)}).iloc[1:3])


def test_concat_of_spilled_views(manager: SpillManager):
    df_base = cudf.DataFrame({"a": range(10)})
    df1, df2 = df_base.iloc[0:1], df_base.iloc[1:3]
    manager.spill_to_device_limit(0)
    assert len(manager.base_buffers()) == 1
    assert gen_df.is_spilled(df_base)

    res = cudf.concat([df1, df2])

    assert len(manager.base_buffers()) == 4
    assert gen_df.is_spilled(df_base)
    assert not gen_df.is_spilled(df1)
    assert not gen_df.is_spilled(df2)
    assert not gen_df.is_spilled(res)


def test_other_buffers(manager: SpillManager):
    buf = Buffer(bytearray(100))
    assert len(manager.other_buffers()) == 1
    assert manager.other_buffers()[0] is buf


def test_ptr_restricted(manager: SpillManager):
    buf = SpillableBuffer(
        data=rmm.DeviceBuffer(size=10), exposed=False, manager=manager
    )
    assert buf.spillable
    assert buf.expose_counter == 1
    spill_lock = SpillLock()
    buf.get_ptr(spill_lock=spill_lock)
    assert not buf.spillable
    assert buf.expose_counter == 2
    buf.get_ptr(spill_lock=spill_lock)
    assert not buf.spillable
    assert buf.expose_counter == 3
    del spill_lock
    assert buf.spillable
    assert buf.expose_counter == 1


def test_expose_statistics(manager: SpillManager):
    manager._expose_statistics = {}  # Enable expose statistics
    assert len(manager.get_expose_statistics()) == 0

    buffers = [
        SpillableBuffer(
            data=rmm.DeviceBuffer(size=10), exposed=False, manager=manager
        )
        for _ in range(10)
    ]

    # Expose the first buffer
    buffers[0].ptr
    assert len(manager.get_expose_statistics()) == 1
    stat = manager.get_expose_statistics()[0]
    assert stat.count == 1
    assert stat.total_nbytes == buffers[0].nbytes
    assert stat.spilled_nbytes == 0

    # Expose all 10 buffers
    for i in range(10):
        buffers[i].ptr

    assert len(manager.get_expose_statistics()) == 2

    # The stats of the first buffer should now be the last
    assert manager.get_expose_statistics()[1] == stat
    # The rest should accumulate to a single stat
    stat = manager.get_expose_statistics()[0]
    assert stat.count == 9
    assert stat.total_nbytes == buffers[0].nbytes * 9
    assert stat.spilled_nbytes == 0

    # Create and spill 10 new buffers
    buffers = [
        SpillableBuffer(
            data=rmm.DeviceBuffer(size=10), exposed=False, manager=manager
        )
        for _ in range(10)
    ]
    manager.spill_to_device_limit(0)

    # Expose the new buffers and check that they are counted as spilled
    for i in range(10):
        buffers[i].ptr
    assert len(manager.get_expose_statistics()) == 3
    stat = manager.get_expose_statistics()[0]
    assert stat.count == 10
    assert stat.total_nbytes == buffers[0].nbytes * 10
    assert stat.spilled_nbytes == buffers[0].nbytes * 10


@pytest.mark.parametrize("target", ["gpu", "cpu"])
def test_serialize_device(manager, target):
    df1 = gen_df(target=target)
    header, frames = df1.device_serialize()
    assert len(frames) == 1
    if target == "gpu":
        assert isinstance(frames[0], Buffer)
        assert gen_df.buffer(df1).expose_counter == 2
    else:
        assert gen_df.buffer(df1).expose_counter == 1
        assert isinstance(frames[0], memoryview)

    df2 = Serializable.device_deserialize(header, frames)
    assert_eq(df1, df2)


@pytest.mark.parametrize("target", ["gpu", "cpu"])
@pytest.mark.parametrize("view", [None, slice(0, 2), slice(1, 3)])
def test_serialize_host(manager, target, view):
    # Unspilled df becomes spilled after host serialization
    df1 = gen_df(target=target)
    if view is not None:
        df1 = df1.iloc[view]
    header, frames = df1.host_serialize()
    assert all(isinstance(f, memoryview) for f in frames)
    df2 = Serializable.host_deserialize(header, frames)
    assert gen_df.is_spilled(df2)
    assert_eq(df1, df2)


def test_serialize_dask_dataframe(manager: SpillManager):
    protocol = pytest.importorskip("distributed.protocol")

    df1 = gen_df(target="gpu")
    header, frames = protocol.serialize(
        df1, serializers=("dask",), on_error="raise"
    )
    buf: SpillableBuffer = gen_df.buffer(df1)
    assert len(frames) == 1
    assert isinstance(frames[0], memoryview)
    # Check that the memoryview and frames is the same memory
    assert (
        np.array(buf.memoryview()).__array_interface__["data"]
        == np.array(frames[0]).__array_interface__["data"]
    )

    df2 = protocol.deserialize(header, frames)
    assert gen_df.is_spilled(df2)
    assert_eq(df1, df2)


def test_serialize_cuda_dataframe(manager: SpillManager):
    protocol = pytest.importorskip("distributed.protocol")

    df1 = gen_df(target="gpu")
    header, frames = protocol.serialize(
        df1, serializers=("cuda",), on_error="raise"
    )
    buf: SpillableBuffer = gen_df.buffer(df1)
    assert buf.expose_counter == 2
    assert len(frames) == 1
    assert isinstance(frames[0], Buffer)
    assert frames[0].ptr == buf.ptr

    df2 = protocol.deserialize(header, frames)
    assert buf.ptr == gen_df.buffer(df2).ptr
    assert_eq(df1, df2)

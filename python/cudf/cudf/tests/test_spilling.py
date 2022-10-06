# Copyright (c) 2022, NVIDIA CORPORATION.


import gc
import warnings
from typing import Tuple

import cupy
import numpy as np
import pandas
import pandas.testing
import pytest

import rmm

import cudf
import cudf.core.buffer.spill_manager
from cudf.core.abc import Serializable
from cudf.core.buffer import Buffer, DeviceBufferLike, as_device_buffer_like
from cudf.core.buffer.spill_manager import (
    SpillManager,
    get_rmm_memory_resource_stack,
    global_manager_get,
    global_manager_reset,
)
from cudf.core.buffer.spillable_buffer import SpillableBuffer, SpillLock
from cudf.core.buffer.utils import (
    get_columns,
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


def spilled_and_unspilled(manager: SpillManager) -> Tuple[int, int]:
    """Get bytes spilled and unspilled known by the manager"""
    spilled = sum(buf.size for buf in manager.base_buffers() if buf.is_spilled)
    unspilled = sum(
        buf.size for buf in manager.base_buffers() if not buf.is_spilled
    )
    return spilled, unspilled


@pytest.fixture
def manager(request):
    """Fixture to enable and make a spilling manager availabe"""
    kwargs = dict(getattr(request, "param", {}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        global_manager_reset(manager=SpillManager(**kwargs))
        yield global_manager_get()
        if request.node.report["call"].failed:
            # Ignore `overwriting non-empty manager` errors when
            # test is failing.
            warnings.simplefilter("ignore")
        global_manager_reset(manager=None)


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


@pytest.mark.parametrize(
    "attribute",
    [
        "ptr",
        "get_ptr",
        "memoryview",
        "is_spilled",
        "exposed",
        "spillable",
        "spill_lock",
        "__spill__",
    ],
)
def test_spillable_buffer_view_attributes(manager: SpillManager, attribute):
    base = SpillableBuffer(
        data=rmm.DeviceBuffer(size=10), exposed=False, manager=manager
    )
    view = base[:]
    attr_base = getattr(base, attribute)
    attr_view = getattr(view, attribute)
    if callable(attr_view):
        pass
    else:
        assert attr_base == attr_view


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
    assert len(df._data._data["x"].data._spill_locks) == 1
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
    def clear():
        # In order to enabling monkey patching of the environment variables
        # mark the global manager as uninitialized.
        global_manager_reset(None)
        cudf.core.buffer.spill_manager._global_manager_uninitialized = True

    clear()
    monkeypatch.setenv("CUDF_SPILL_ON_DEMAND", "off")
    monkeypatch.setenv("CUDF_SPILL", "off")
    assert global_manager_get() is None

    clear()
    monkeypatch.setenv("CUDF_SPILL", "on")
    manager = global_manager_get()
    assert isinstance(manager, SpillManager)
    assert manager._spill_on_demand is False
    assert manager._device_memory_limit is None

    clear()
    monkeypatch.setenv("CUDF_SPILL_DEVICE_LIMIT", "1000")
    monkeypatch.setenv("CUDF_SPILL_STAT_EXPOSE", "on")
    manager = global_manager_get()
    assert isinstance(manager, SpillManager)
    assert manager._device_memory_limit == 1000
    assert isinstance(manager._expose_statistics, dict)


def test_spill_device_memory(manager: SpillManager):
    df = gen_df()
    assert spilled_and_unspilled(manager) == (0, gen_df.buffer_size)
    manager.spill_device_memory()
    assert spilled_and_unspilled(manager) == (gen_df.buffer_size, 0)
    del df
    assert spilled_and_unspilled(manager) == (0, 0)
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
    assert spilled_and_unspilled(manager) == (0, gen_df.buffer_size * 2)
    manager.spill_to_device_limit(device_limit=0)
    assert spilled_and_unspilled(manager) == (gen_df.buffer_size * 2, 0)
    df3 = df1 + df2
    manager.spill_to_device_limit(device_limit=0)
    assert spilled_and_unspilled(manager) == (gen_df.buffer_size * 3, 0)
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
    assert spilled_and_unspilled(manager) == (gen_df.buffer_size * 2, 0)
    df1 + df2
    # Notice, while performing the addintion both df1 and df2 are unspillable
    assert spilled_and_unspilled(manager) == (0, gen_df.buffer_size * 2)
    manager.spill_to_device_limit()
    assert spilled_and_unspilled(manager) == (gen_df.buffer_size * 2, 0)


def test_lookup_address_range(manager: SpillManager):
    df = gen_df()
    buf = gen_df.buffer(df)
    buffers = manager.base_buffers()
    assert len(buffers) == 1
    (buf,) = buffers
    assert gen_df.buffer(df) is buf
    assert manager.lookup_address_range(buf.ptr, buf.size)[0] is buf
    assert manager.lookup_address_range(buf.ptr + 1, buf.size - 1)[0] is buf
    assert manager.lookup_address_range(buf.ptr + 1, buf.size + 1)[0] is buf
    assert manager.lookup_address_range(buf.ptr - 1, buf.size - 1)[0] is buf
    assert manager.lookup_address_range(buf.ptr - 1, buf.size + 1)[0] is buf
    assert not manager.lookup_address_range(buf.ptr + buf.size, buf.size)
    assert not manager.lookup_address_range(buf.ptr - buf.size, buf.size)


def test_external_memory_never_spills(manager):
    """
    Test that external data, i.e., data not managed by RMM,
    is never spilled
    """

    cupy.cuda.set_allocator()  # uses default allocator

    a = cupy.asarray([1, 2, 3])
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


def test_ptr_restricted(manager: SpillManager):
    buf = SpillableBuffer(
        data=rmm.DeviceBuffer(size=10), exposed=False, manager=manager
    )
    assert buf.spillable
    assert len(buf._spill_locks) == 0
    slock1 = SpillLock()
    buf.get_ptr(spill_lock=slock1)
    assert not buf.spillable
    assert len(buf._spill_locks) == 1
    slock2 = buf.spill_lock()
    buf.get_ptr(spill_lock=slock2)
    assert not buf.spillable
    assert len(buf._spill_locks) == 2
    del slock1
    assert len(buf._spill_locks) == 1
    del slock2
    assert len(buf._spill_locks) == 0
    assert buf.spillable


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
@pytest.mark.parametrize("view", [None, slice(0, 2), slice(1, 3)])
def test_serialize_device(manager, target, view):
    df1 = gen_df()
    if view is not None:
        df1 = df1.iloc[view]
    gen_df.buffer(df1).__spill__(target=target)

    header, frames = df1.device_serialize()
    assert len(frames) == 1
    if target == "gpu":
        assert isinstance(frames[0], Buffer)
        assert not gen_df.is_spilled(df1)
        assert not gen_df.is_spillable(df1)
        frames[0] = cupy.array(frames[0], copy=True)
    else:
        assert isinstance(frames[0], memoryview)
        assert gen_df.is_spilled(df1)
        assert gen_df.is_spillable(df1)

    df2 = Serializable.device_deserialize(header, frames)
    assert_eq(df1, df2)


@pytest.mark.parametrize("target", ["gpu", "cpu"])
@pytest.mark.parametrize("view", [None, slice(0, 2), slice(1, 3)])
def test_serialize_host(manager, target, view):
    df1 = gen_df()
    if view is not None:
        df1 = df1.iloc[view]
    gen_df.buffer(df1).__spill__(target=target)

    # Unspilled df becomes spilled after host serialization
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
    assert len(buf._spill_locks) == 1
    assert len(frames) == 1
    assert isinstance(frames[0], Buffer)
    assert frames[0].ptr == buf.ptr

    frames[0] = cupy.array(frames[0], copy=True)
    df2 = protocol.deserialize(header, frames)
    assert_eq(df1, df2)


def test_get_rmm_memory_resource_stack():
    mr1 = rmm.mr.get_current_device_resource()
    assert all(
        not isinstance(m, rmm.mr.FailureCallbackResourceAdaptor)
        for m in get_rmm_memory_resource_stack(mr1)
    )

    mr2 = rmm.mr.FailureCallbackResourceAdaptor(mr1, lambda x: False)
    assert get_rmm_memory_resource_stack(mr2)[0] is mr2
    assert get_rmm_memory_resource_stack(mr2)[1] is mr1

    mr3 = rmm.mr.FixedSizeMemoryResource(mr2)
    assert get_rmm_memory_resource_stack(mr3)[0] is mr3
    assert get_rmm_memory_resource_stack(mr3)[1] is mr2
    assert get_rmm_memory_resource_stack(mr3)[2] is mr1

    mr4 = rmm.mr.FailureCallbackResourceAdaptor(mr3, lambda x: False)
    assert get_rmm_memory_resource_stack(mr4)[0] is mr4
    assert get_rmm_memory_resource_stack(mr4)[1] is mr3
    assert get_rmm_memory_resource_stack(mr4)[2] is mr2
    assert get_rmm_memory_resource_stack(mr4)[3] is mr1

# Copyright (c) 2022, NVIDIA CORPORATION.

import importlib
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import cupy
import numpy as np
import pandas
import pandas.testing
import pytest

import rmm

import cudf
import cudf.core.buffer.spill_manager
import cudf.options
from cudf.core.abc import Serializable
from cudf.core.buffer import (
    Buffer,
    acquire_spill_lock,
    as_buffer,
    get_spill_lock,
)
from cudf.core.buffer.spill_manager import (
    SpillManager,
    get_global_manager,
    get_rmm_memory_resource_stack,
    set_global_manager,
)
from cudf.core.buffer.spillable_buffer import (
    SpillableBuffer,
    SpillableBufferSlice,
    SpillLock,
)
from cudf.testing._utils import assert_eq

if get_global_manager() is not None:
    pytest.skip(
        "cannot test spilling when enabled globally, set `CUDF_SPILL=off`",
        allow_module_level=True,
    )


def gen_df(target="gpu") -> cudf.DataFrame:
    ret = cudf.DataFrame({"a": [1, 2, 3]})
    if target != "gpu":
        gen_df.buffer(ret).spill(target=target)
    return ret


gen_df.buffer = lambda df: df._data._data["a"].data
gen_df.is_spilled = lambda df: gen_df.buffer(df).is_spilled
gen_df.is_spillable = lambda df: gen_df.buffer(df).spillable
gen_df.buffer_size = gen_df.buffer(gen_df()).size


def spilled_and_unspilled(manager: SpillManager) -> Tuple[int, int]:
    """Get bytes spilled and unspilled known by the manager"""
    spilled = sum(buf.size for buf in manager.buffers() if buf.is_spilled)
    unspilled = sum(
        buf.size for buf in manager.buffers() if not buf.is_spilled
    )
    return spilled, unspilled


@pytest.fixture
def manager(request):
    """Fixture to enable and make a spilling manager availabe"""
    kwargs = dict(getattr(request, "param", {}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        set_global_manager(manager=SpillManager(**kwargs))
        yield get_global_manager()
        # Retrieving the test result using the `pytest_runtest_makereport`
        # hook from conftest.py
        if request.node.report["call"].failed:
            # Ignore `overwriting non-empty manager` errors when
            # test is failing.
            warnings.simplefilter("ignore")
        set_global_manager(manager=None)


def test_spillable_buffer(manager: SpillManager):
    buf = as_buffer(data=rmm.DeviceBuffer(size=10), exposed=False)
    assert isinstance(buf, SpillableBuffer)
    assert buf.spillable
    buf.ptr  # Expose pointer
    assert buf.exposed
    assert not buf.spillable
    buf = as_buffer(data=rmm.DeviceBuffer(size=10), exposed=False)
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
        "spill",
    ],
)
def test_spillable_buffer_view_attributes(manager: SpillManager, attribute):
    base = as_buffer(data=rmm.DeviceBuffer(size=10), exposed=False)
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
    assert len(df._data._data["x"].base_data._spill_locks) == 0
    gb._groupby
    # `gb._groupby`, which is cached on `gb`, holds a spill lock
    assert len(df._data._data["x"].base_data._spill_locks) == 1
    assert not df._data._data["x"].data.spillable
    del gb
    assert df._data._data["x"].data.spillable


def test_spilling_buffer(manager: SpillManager):
    buf = as_buffer(rmm.DeviceBuffer(size=10), exposed=False)
    buf.spill(target="cpu")
    assert buf.is_spilled
    buf.ptr  # Expose pointer and trigger unspill
    assert not buf.is_spilled
    with pytest.raises(ValueError, match="unspillable buffer"):
        buf.spill(target="cpu")


def test_environment_variables(monkeypatch):
    def reload_options():
        # In order to enabling monkey patching of the environment variables
        # mark the global manager as uninitialized.
        set_global_manager(None)
        cudf.core.buffer.spill_manager._global_manager_uninitialized = True
        importlib.reload(cudf.options)

    monkeypatch.setenv("CUDF_SPILL_ON_DEMAND", "off")
    monkeypatch.setenv("CUDF_SPILL", "off")
    reload_options()
    assert get_global_manager() is None

    monkeypatch.setenv("CUDF_SPILL", "on")
    reload_options()
    manager = get_global_manager()
    assert isinstance(manager, SpillManager)
    assert manager._spill_on_demand is False
    assert manager._device_memory_limit is None

    monkeypatch.setenv("CUDF_SPILL_DEVICE_LIMIT", "1000")
    reload_options()
    manager = get_global_manager()
    assert isinstance(manager, SpillManager)
    assert manager._device_memory_limit == 1000


def test_spill_device_memory(manager: SpillManager):
    df = gen_df()
    assert spilled_and_unspilled(manager) == (0, gen_df.buffer_size)
    manager.spill_device_memory(nbytes=1)
    assert spilled_and_unspilled(manager) == (gen_df.buffer_size, 0)
    del df
    assert spilled_and_unspilled(manager) == (0, 0)
    df1 = gen_df()
    df2 = gen_df()
    manager.spill_device_memory(nbytes=1)
    assert gen_df.is_spilled(df1)
    assert not gen_df.is_spilled(df2)
    manager.spill_device_memory(nbytes=1)
    assert gen_df.is_spilled(df1)
    assert gen_df.is_spilled(df2)
    df3 = df1 + df2
    assert not gen_df.is_spilled(df1)
    assert not gen_df.is_spilled(df2)
    assert not gen_df.is_spilled(df3)
    manager.spill_device_memory(nbytes=1)
    assert gen_df.is_spilled(df1)
    assert not gen_df.is_spilled(df2)
    assert not gen_df.is_spilled(df3)
    df2.abs()  # Should change the access time
    manager.spill_device_memory(nbytes=1)
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


def test_external_memory_never_spills(manager):
    """
    Test that external data, i.e., data not managed by RMM,
    is never spilled
    """

    cupy.cuda.set_allocator()  # uses default allocator

    a = cupy.asarray([1, 2, 3])
    s = cudf.Series(a)
    assert len(manager.buffers()) == 0
    assert not s._data[None].data.spillable


def test_spilling_df_views(manager):
    df = gen_df(target="cpu")
    assert gen_df.is_spilled(df)
    df_view = df.loc[1:]
    assert gen_df.is_spillable(df_view)
    assert gen_df.is_spillable(df)


def test_modify_spilled_views(manager):
    df = gen_df()
    df_view = df.iloc[1:]
    buf = gen_df.buffer(df)
    buf.spill(target="cpu")

    # modify the spilled df and check that the changes are reflected
    # in the view
    df.iloc[1:] = 0
    assert_eq(df_view, df.iloc[1:])

    # now, modify the view and check that the changes are reflected in
    # the df
    df_view.iloc[:] = -1
    assert_eq(df_view, df.iloc[1:])


def test_ptr_restricted(manager: SpillManager):
    buf = as_buffer(data=rmm.DeviceBuffer(size=10), exposed=False)
    assert buf.spillable
    assert len(buf._spill_locks) == 0
    slock1 = SpillLock()
    buf.get_ptr(spill_lock=slock1)
    assert not buf.spillable
    assert len(buf._spill_locks) == 1
    slock2 = SpillLock()
    buf.spill_lock(spill_lock=slock2)
    buf.get_ptr(spill_lock=slock2)
    assert not buf.spillable
    assert len(buf._spill_locks) == 2
    del slock1
    assert len(buf._spill_locks) == 1
    del slock2
    assert len(buf._spill_locks) == 0
    assert buf.spillable


def test_get_spill_lock(manager: SpillManager):
    @acquire_spill_lock()
    def f(sleep=False, nest=0):
        if sleep:
            time.sleep(random.random() / 100)
        if nest:
            return f(nest=nest - 1)
        return get_spill_lock()

    assert get_spill_lock() is None
    slock = f()
    assert isinstance(slock, SpillLock)
    assert get_spill_lock() is None
    slock = f(nest=2)
    assert isinstance(slock, SpillLock)
    assert get_spill_lock() is None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures_with_spill_lock = []
        futures_without_spill_lock = []
        for _ in range(100):
            futures_with_spill_lock.append(
                executor.submit(f, sleep=True, nest=1)
            )
            futures_without_spill_lock.append(
                executor.submit(f, sleep=True, nest=1)
            )
        all(isinstance(f.result(), SpillLock) for f in futures_with_spill_lock)
        all(f is None for f in futures_without_spill_lock)


def test_get_spill_lock_no_manager():
    """When spilling is disabled, get_spill_lock() should return None always"""

    @acquire_spill_lock()
    def f():
        return get_spill_lock()

    assert get_spill_lock() is None
    assert f() is None


@pytest.mark.parametrize("target", ["gpu", "cpu"])
@pytest.mark.parametrize("view", [None, slice(0, 2), slice(1, 3)])
def test_serialize_device(manager, target, view):
    df1 = gen_df()
    if view is not None:
        df1 = df1.iloc[view]
    gen_df.buffer(df1).spill(target=target)

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
    gen_df.buffer(df1).spill(target=target)

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
    buf: SpillableBufferSlice = gen_df.buffer(df1)
    assert len(buf._base._spill_locks) == 1
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


def test_df_transpose(manager: SpillManager):
    df1 = cudf.DataFrame({"x": [1, 2]})
    df2 = df1.transpose()
    # For now, all buffers are marked as exposed
    assert df1._data._data["x"].data.exposed
    assert df2._data._data[0].data.exposed
    assert df2._data._data[1].data.exposed


@pytest.mark.parametrize("dtype", ["uint8", "uint64"])
def test_memoryview_slice(manager: SpillManager, dtype):
    """Check .memoryview() of a sliced spillable buffer"""

    data = np.arange(10, dtype=dtype)
    # memoryview of a sliced spillable buffer
    m1 = as_buffer(data=data)[1:-1].memoryview()
    # sliced memoryview of data as bytes
    m2 = memoryview(data).cast("B")[1:-1]
    assert m1 == m2

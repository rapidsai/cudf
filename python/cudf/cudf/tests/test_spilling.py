# Copyright (c) 2022-2023, NVIDIA CORPORATION.

import importlib
import random
import time
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

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


def single_column_df(target="gpu") -> cudf.DataFrame:
    """Create a standard single column dataframe used for testing

    Use `single_column_df_data`, `single_column_df_base_data`,
    `gen_df_data_nbytes` for easy access to the buffer of the column.

    Notice, this is just for convenience, there is nothing special
    about this dataframe.

    Parameters
    ----------
    target : str, optional
        Set the spill state of the dataframe

    Return
    ------
    DataFrame
        A standard dataframe with a single column
    """
    ret = cudf.DataFrame({"a": [1, 2, 3]})
    if target != "gpu":
        single_column_df_data(ret).spill(target=target)
    return ret


def single_column_df_data(df: cudf.DataFrame) -> SpillableBuffer:
    """Access `.data` of the column of a standard dataframe"""
    ret = df._data._data["a"].data
    assert isinstance(ret, SpillableBuffer)
    return ret


def single_column_df_base_data(df: cudf.DataFrame) -> SpillableBuffer:
    """Access `.base_data` of the column of a standard dataframe"""
    ret = df._data._data["a"].base_data
    assert isinstance(ret, SpillableBuffer)
    return ret


# Get number of bytes of the column of a standard dataframe
gen_df_data_nbytes = single_column_df()._data._data["a"].data.nbytes


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
    buf.mark_exposed()
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
        "get_ptr",
        "memoryview",
        "is_spilled",
        "exposed",
        "spillable",
        "spill_lock",
        "spill",
        "memory_info",
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


@pytest.mark.parametrize("target", ["gpu", "cpu"])
def test_memory_info(manager: SpillManager, target):
    if target == "gpu":
        mem = rmm.DeviceBuffer(size=10)
        ptr = mem.ptr
    elif target == "cpu":
        mem = np.empty(10, dtype="u1")
        ptr = mem.__array_interface__["data"][0]
    b = as_buffer(data=mem, exposed=False)
    assert b.memory_info() == (ptr, mem.size, target)
    assert b[:].memory_info() == (ptr, mem.size, target)
    assert b[:-1].memory_info() == (ptr, mem.size - 1, target)
    assert b[1:].memory_info() == (ptr + 1, mem.size - 1, target)
    assert b[2:4].memory_info() == (ptr + 2, 2, target)


def test_from_pandas(manager: SpillManager):
    pdf1 = pandas.DataFrame({"a": [1, 2, 3]})
    df = cudf.from_pandas(pdf1)
    assert single_column_df_data(df).spillable
    pdf2 = df.to_pandas()
    pandas.testing.assert_frame_equal(pdf1, pdf2)


def test_creations(manager: SpillManager):
    df = single_column_df()
    assert single_column_df_data(df).spillable

    df = cudf.datasets.timeseries(dtypes={"a": float})
    assert single_column_df_data(df).spillable

    df = cudf.datasets.randomdata(dtypes={"a": float})
    assert single_column_df_data(df).spillable


def test_spillable_df_groupby(manager: SpillManager):
    df = cudf.DataFrame({"a": [1, 1, 1]})
    gb = df.groupby("a")
    assert len(single_column_df_base_data(df)._spill_locks) == 0
    gb._groupby
    # `gb._groupby`, which is cached on `gb`, holds a spill lock
    assert len(single_column_df_base_data(df)._spill_locks) == 1
    assert not single_column_df_data(df).spillable
    del gb
    assert single_column_df_data(df).spillable


def test_spilling_buffer(manager: SpillManager):
    buf = as_buffer(rmm.DeviceBuffer(size=10), exposed=False)
    buf.spill(target="cpu")
    assert buf.is_spilled
    buf.mark_exposed()  # Expose pointer and trigger unspill
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
    assert manager.statistics.level == 0

    monkeypatch.setenv("CUDF_SPILL_DEVICE_LIMIT", "1000")
    reload_options()
    manager = get_global_manager()
    assert isinstance(manager, SpillManager)
    assert manager._device_memory_limit == 1000
    assert manager.statistics.level == 0

    monkeypatch.setenv("CUDF_SPILL_STATS", "1")
    reload_options()
    manager = get_global_manager()
    assert isinstance(manager, SpillManager)
    assert manager.statistics.level == 1

    monkeypatch.setenv("CUDF_SPILL_STATS", "2")
    reload_options()
    manager = get_global_manager()
    assert isinstance(manager, SpillManager)
    assert manager.statistics.level == 2


def test_spill_device_memory(manager: SpillManager):
    df = single_column_df()
    assert spilled_and_unspilled(manager) == (0, gen_df_data_nbytes)
    manager.spill_device_memory(nbytes=1)
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes, 0)
    del df
    assert spilled_and_unspilled(manager) == (0, 0)
    df1 = single_column_df()
    df2 = single_column_df()
    manager.spill_device_memory(nbytes=1)
    assert single_column_df_data(df1).is_spilled
    assert not single_column_df_data(df2).is_spilled
    manager.spill_device_memory(nbytes=1)
    assert single_column_df_data(df1).is_spilled
    assert single_column_df_data(df2).is_spilled
    df3 = df1 + df2
    assert not single_column_df_data(df1).is_spilled
    assert not single_column_df_data(df2).is_spilled
    assert not single_column_df_data(df3).is_spilled
    manager.spill_device_memory(nbytes=1)
    assert single_column_df_data(df1).is_spilled
    assert not single_column_df_data(df2).is_spilled
    assert not single_column_df_data(df3).is_spilled
    df2.abs()  # Should change the access time
    manager.spill_device_memory(nbytes=1)
    assert single_column_df_data(df1).is_spilled
    assert not single_column_df_data(df2).is_spilled
    assert single_column_df_data(df3).is_spilled


def test_spill_to_device_limit(manager: SpillManager):
    df1 = single_column_df()
    df2 = single_column_df()
    assert spilled_and_unspilled(manager) == (0, gen_df_data_nbytes * 2)
    manager.spill_to_device_limit(device_limit=0)
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes * 2, 0)
    df3 = df1 + df2
    manager.spill_to_device_limit(device_limit=0)
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes * 3, 0)
    assert single_column_df_data(df1).is_spilled
    assert single_column_df_data(df2).is_spilled
    assert single_column_df_data(df3).is_spilled


@pytest.mark.parametrize(
    "manager", [{"device_memory_limit": 0}], indirect=True
)
def test_zero_device_limit(manager: SpillManager):
    assert manager._device_memory_limit == 0
    df1 = single_column_df()
    df2 = single_column_df()
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes * 2, 0)
    df1 + df2
    # Notice, while performing the addintion both df1 and df2 are unspillable
    assert spilled_and_unspilled(manager) == (0, gen_df_data_nbytes * 2)
    manager.spill_to_device_limit()
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes * 2, 0)


def test_spill_df_index(manager: SpillManager):
    df = single_column_df()
    df.index = [1, 3, 2]  # use a materialized index
    assert spilled_and_unspilled(manager) == (0, gen_df_data_nbytes * 2)

    manager.spill_to_device_limit(gen_df_data_nbytes)
    assert spilled_and_unspilled(manager) == (
        gen_df_data_nbytes,
        gen_df_data_nbytes,
    )

    manager.spill_to_device_limit(0)
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes * 2, 0)


def test_external_memory(manager):
    cupy.cuda.set_allocator()  # uses default allocator
    cpy = cupy.asarray([1, 2, 3])
    s = cudf.Series(cpy)
    # Check that the cupy array is still alive after overwriting `cpy`
    cpy = weakref.ref(cpy)
    assert cpy() is not None
    # Check that the series is spillable and known by the spill manager
    assert len(manager.buffers()) == 1
    assert s._data[None].data.spillable


def test_spilling_df_views(manager):
    df = single_column_df(target="cpu")
    assert single_column_df_data(df).is_spilled
    df_view = df.loc[1:]
    assert single_column_df_data(df_view).spillable
    assert single_column_df_data(df).spillable


def test_modify_spilled_views(manager):
    df = single_column_df()
    df_view = df.iloc[1:]
    buf = single_column_df_data(df)
    buf.spill(target="cpu")

    # modify the spilled df and check that the changes are reflected
    # in the view
    df.iloc[1:] = 0
    assert_eq(df_view, df.iloc[1:])

    # now, modify the view and check that the changes are reflected in
    # the df
    df_view.iloc[:] = -1
    assert_eq(df_view, df.iloc[1:])


@pytest.mark.parametrize("target", ["gpu", "cpu"])
def test_get_ptr(manager: SpillManager, target):
    if target == "gpu":
        mem = rmm.DeviceBuffer(size=10)
    elif target == "cpu":
        mem = np.empty(10, dtype="u1")
    buf = as_buffer(data=mem, exposed=False)
    assert buf.spillable
    assert len(buf._spill_locks) == 0
    with acquire_spill_lock():
        buf.get_ptr(mode="read")
        assert not buf.spillable
        with acquire_spill_lock():
            buf.get_ptr(mode="read")
            assert not buf.spillable
        assert not buf.spillable
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
    df1 = single_column_df()
    if view is not None:
        df1 = df1.iloc[view]
    single_column_df_data(df1).spill(target=target)

    header, frames = df1.device_serialize()
    assert len(frames) == 1
    if target == "gpu":
        assert isinstance(frames[0], Buffer)
        assert not single_column_df_data(df1).is_spilled
        assert not single_column_df_data(df1).spillable
        frames[0] = cupy.array(frames[0], copy=True)
    else:
        assert isinstance(frames[0], memoryview)
        assert single_column_df_data(df1).is_spilled
        assert single_column_df_data(df1).spillable

    df2 = Serializable.device_deserialize(header, frames)
    assert_eq(df1, df2)


@pytest.mark.parametrize("target", ["gpu", "cpu"])
@pytest.mark.parametrize("view", [None, slice(0, 2), slice(1, 3)])
def test_serialize_host(manager, target, view):
    df1 = single_column_df()
    if view is not None:
        df1 = df1.iloc[view]
    single_column_df_data(df1).spill(target=target)

    # Unspilled df becomes spilled after host serialization
    header, frames = df1.host_serialize()
    assert all(isinstance(f, memoryview) for f in frames)
    df2 = Serializable.host_deserialize(header, frames)
    assert single_column_df_data(df2).is_spilled
    assert_eq(df1, df2)


def test_serialize_dask_dataframe(manager: SpillManager):
    protocol = pytest.importorskip("distributed.protocol")

    df1 = single_column_df(target="gpu")
    header, frames = protocol.serialize(
        df1, serializers=("dask",), on_error="raise"
    )
    buf = single_column_df_data(df1)
    assert len(frames) == 1
    assert isinstance(frames[0], memoryview)
    # Check that the memoryview and frames is the same memory
    assert (
        np.array(buf.memoryview()).__array_interface__["data"]
        == np.array(frames[0]).__array_interface__["data"]
    )

    df2 = protocol.deserialize(header, frames)
    assert single_column_df_data(df2).is_spilled
    assert_eq(df1, df2)


def test_serialize_cuda_dataframe(manager: SpillManager):
    protocol = pytest.importorskip("distributed.protocol")

    df1 = single_column_df(target="gpu")
    header, frames = protocol.serialize(
        df1, serializers=("cuda",), on_error="raise"
    )
    buf: SpillableBufferSlice = single_column_df_data(df1)
    assert len(buf._base._spill_locks) == 1
    assert len(frames) == 1
    assert isinstance(frames[0], Buffer)
    assert frames[0].get_ptr(mode="read") == buf.get_ptr(mode="read")

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
    df1 = cudf.DataFrame({"a": [1, 2]})
    df2 = df1.transpose()
    # For now, all buffers are marked as exposed
    assert df1._data._data["a"].data.exposed
    assert df2._data._data[0].data.exposed
    assert df2._data._data[1].data.exposed


def test_as_buffer_of_spillable_buffer(manager: SpillManager):
    data = cupy.arange(10, dtype="u1")
    b1 = as_buffer(data, exposed=False)
    assert isinstance(b1, SpillableBuffer)
    assert b1.owner is data
    b2 = as_buffer(b1)
    assert b1 is b2

    with pytest.raises(
        ValueError,
        match="buffer must either be exposed or spilled locked",
    ):
        # Use `memory_info` to access device point _without_ making
        # the buffer unspillable.
        b3 = as_buffer(b1.memory_info()[0], size=b1.size, owner=b1)

    with acquire_spill_lock():
        b3 = as_buffer(b1.get_ptr(mode="read"), size=b1.size, owner=b1)
    assert isinstance(b3, SpillableBufferSlice)
    assert b3.owner is b1

    b4 = as_buffer(
        b1.get_ptr(mode="write") + data.itemsize,
        size=b1.size - data.itemsize,
        owner=b3,
    )
    assert isinstance(b4, SpillableBufferSlice)
    assert b4.owner is b1
    assert all(cupy.array(b4.memoryview()) == data[1:])

    b5 = as_buffer(b4.get_ptr(mode="write"), size=b4.size - 1, owner=b4)
    assert isinstance(b5, SpillableBufferSlice)
    assert b5.owner is b1
    assert all(cupy.array(b5.memoryview()) == data[1:-1])


@pytest.mark.parametrize("dtype", ["uint8", "uint64"])
def test_memoryview_slice(manager: SpillManager, dtype):
    """Check .memoryview() of a sliced spillable buffer"""

    data = np.arange(10, dtype=dtype)
    # memoryview of a sliced spillable buffer
    m1 = as_buffer(data=data)[1:-1].memoryview()
    # sliced memoryview of data as bytes
    m2 = memoryview(data).cast("B")[1:-1]
    assert m1 == m2


@pytest.mark.parametrize(
    "manager", [{"statistic_level": 0}, {"statistic_level": 1}], indirect=True
)
def test_statistics(manager: SpillManager):
    assert len(manager.statistics.spill_totals) == 0

    buf: SpillableBuffer = as_buffer(
        data=rmm.DeviceBuffer(size=10), exposed=False
    )
    buf.spill(target="cpu")

    if manager.statistics.level == 0:
        assert len(manager.statistics.spill_totals) == 0
        return

    assert len(manager.statistics.spill_totals) == 1
    nbytes, time = manager.statistics.spill_totals[("gpu", "cpu")]
    assert nbytes == buf.size
    assert time > 0

    buf.spill(target="gpu")
    assert len(manager.statistics.spill_totals) == 2
    nbytes, time = manager.statistics.spill_totals[("cpu", "gpu")]
    assert nbytes == buf.size
    assert time > 0


@pytest.mark.parametrize("manager", [{"statistic_level": 2}], indirect=True)
def test_statistics_expose(manager: SpillManager):
    assert len(manager.statistics.spill_totals) == 0

    buffers: List[SpillableBuffer] = [
        as_buffer(data=rmm.DeviceBuffer(size=10), exposed=False)
        for _ in range(10)
    ]

    # Expose the first buffer
    buffers[0].mark_exposed()
    assert len(manager.statistics.exposes) == 1
    stat = list(manager.statistics.exposes.values())[0]
    assert stat.count == 1
    assert stat.total_nbytes == buffers[0].nbytes
    assert stat.spilled_nbytes == 0

    # Expose all 10 buffers
    for i in range(10):
        buffers[i].mark_exposed()

    # The rest of the ptr accesses should accumulate to a single stat
    # because they resolve to the same traceback.
    assert len(manager.statistics.exposes) == 2
    stat = list(manager.statistics.exposes.values())[1]
    assert stat.count == 9
    assert stat.total_nbytes == buffers[0].nbytes * 9
    assert stat.spilled_nbytes == 0

    # Create and spill 10 new buffers
    buffers: List[SpillableBuffer] = [
        as_buffer(data=rmm.DeviceBuffer(size=10), exposed=False)
        for _ in range(10)
    ]

    manager.spill_to_device_limit(0)

    # Expose the new buffers and check that they are counted as spilled
    for i in range(10):
        buffers[i].mark_exposed()
    assert len(manager.statistics.exposes) == 3
    stat = list(manager.statistics.exposes.values())[2]
    assert stat.count == 10
    assert stat.total_nbytes == buffers[0].nbytes * 10
    assert stat.spilled_nbytes == buffers[0].nbytes * 10

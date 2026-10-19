# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pylibcudf as plc
import rmm
from cuda.bindings import driver, runtime

mode = sys.argv[1]
UINT64_MAX = 2**64 - 1


def check(result):
    err = result[0]
    assert err == runtime.cudaError_t.cudaSuccess, err
    return result[1:]


def raw_pool(
    hw_decompress, release_threshold_max, sync_before_destroy, write=False
):
    props = runtime.cudaMemPoolProps()
    props.allocType = runtime.cudaMemAllocationType.cudaMemAllocationTypePinned
    props.location.type = runtime.cudaMemLocationType.cudaMemLocationTypeDevice
    props.location.id = 0
    if hw_decompress:
        props.usage = runtime.cudaMemPoolCreateUsageHwDecompress
    (pool,) = check(runtime.cudaMemPoolCreate(props))
    if release_threshold_max:
        check(
            runtime.cudaMemPoolSetAttribute(
                pool,
                runtime.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold,
                driver.cuuint64_t(UINT64_MAX),
            )
        )
    sizes = (8, 64, 256, 4096)
    ptrs = [
        check(runtime.cudaMallocFromPoolAsync(size, pool, 0))[0]
        for size in sizes
    ]
    if write:
        host = bytes(range(256)) * 16
        for ptr, size in zip(ptrs, sizes, strict=True):
            check(runtime.cudaMemsetAsync(ptr, 0, size, 0))
            check(
                runtime.cudaMemcpyAsync(
                    ptr,
                    host,
                    size,
                    runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    0,
                )
            )
    for ptr in ptrs:
        check(runtime.cudaFreeAsync(ptr, 0))
    if sync_before_destroy:
        check(runtime.cudaDeviceSynchronize())
    check(runtime.cudaMemPoolDestroy(pool))


def rmm_async(sync_before_destroy):
    previous = rmm.mr.get_current_device_resource()
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    table = plc.Table.from_arrow(pa.Table.from_pydict({"a": [1]}))
    rmm.mr.set_current_device_resource(previous)
    del table
    if sync_before_destroy:
        check(runtime.cudaDeviceSynchronize())
    del mr


def rmm_device_buffer_only():
    previous = rmm.mr.get_current_device_resource()
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    buffers = [
        rmm.DeviceBuffer.to_device(bytes(range(256)) * 16) for _ in range(4)
    ]
    rmm.mr.set_current_device_resource(previous)
    del buffers
    del mr


if mode == "rmm-async-destroy-pending":
    rmm_async(sync_before_destroy=False)
elif mode == "rmm-async-destroy-synced":
    rmm_async(sync_before_destroy=True)
elif mode == "raw-hw-max-threshold-pending":
    raw_pool(True, True, sync_before_destroy=False)
elif mode == "raw-hw-max-threshold-synced":
    raw_pool(True, True, sync_before_destroy=True)
elif mode == "raw-plain-max-threshold-pending":
    raw_pool(False, True, sync_before_destroy=False)
elif mode == "raw-hw-max-threshold-written-pending":
    raw_pool(True, True, sync_before_destroy=False, write=True)
elif mode == "raw-plain-max-threshold-written-pending":
    raw_pool(False, True, sync_before_destroy=False, write=True)
elif mode == "rmm-device-buffer-destroy-pending":
    rmm_device_buffer_only()
elif mode != "none":
    raise ValueError(mode)

tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "a.parquet")
pq.write_table(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}), path)
for _ in range(5):
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([path])
    ).build()
    plc.io.parquet.read_parquet(options)
print(f"OK {mode}")

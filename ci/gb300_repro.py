# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pylibcudf as plc
import rmm
from cuda.bindings import runtime

mode = sys.argv[1]
keepalive = []


def check(result):
    err = result[0]
    assert err == runtime.cudaError_t.cudaSuccess, err
    return result[1:]


def raw_pool_alloc_destroy(usage, sync_before_destroy=True):
    props = runtime.cudaMemPoolProps()
    props.allocType = runtime.cudaMemAllocationType.cudaMemAllocationTypePinned
    props.location.type = runtime.cudaMemLocationType.cudaMemLocationTypeDevice
    props.location.id = 0
    props.usage = usage
    (pool,) = check(runtime.cudaMemPoolCreate(props))
    (ptr,) = check(runtime.cudaMallocFromPoolAsync(1 << 20, pool, 0))
    check(runtime.cudaFreeAsync(ptr, 0))
    if sync_before_destroy:
        check(runtime.cudaStreamSynchronize(0))
    check(runtime.cudaMemPoolDestroy(pool))


def rmm_async(keep, allocate=True, sync_before_destroy=False):
    previous = rmm.mr.get_current_device_resource()
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    if allocate:
        table = plc.Table.from_arrow(pa.Table.from_pydict({"a": [1]}))
    rmm.mr.set_current_device_resource(previous)
    if allocate:
        del table
    if sync_before_destroy:
        check(runtime.cudaDeviceSynchronize())
    if keep:
        keepalive.append(mr)
    del mr


if mode == "rmm-async-destroy":
    rmm_async(keep=False)
elif mode == "rmm-async-sync-before-destroy":
    rmm_async(keep=False, sync_before_destroy=True)
elif mode == "rmm-async-noalloc-destroy":
    rmm_async(keep=False, allocate=False)
elif mode == "raw-hw-pool-destroy-pending":
    raw_pool_alloc_destroy(
        runtime.cudaMemPoolCreateUsageHwDecompress, sync_before_destroy=False
    )
elif mode == "raw-plain-pool-destroy-pending":
    raw_pool_alloc_destroy(0, sync_before_destroy=False)

tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "a.parquet")
pq.write_table(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}), path)
for _ in range(5):
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([path])
    ).build()
    plc.io.parquet.read_parquet(options)
print(f"OK {mode}")

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


def raw_pool_alloc_destroy(usage):
    props = runtime.cudaMemPoolProps()
    props.allocType = runtime.cudaMemAllocationType.cudaMemAllocationTypePinned
    props.location.type = runtime.cudaMemLocationType.cudaMemLocationTypeDevice
    props.location.id = 0
    props.usage = usage
    (pool,) = check(runtime.cudaMemPoolCreate(props))
    (ptr,) = check(runtime.cudaMallocFromPoolAsync(1 << 20, pool, 0))
    check(runtime.cudaFreeAsync(ptr, 0))
    check(runtime.cudaStreamSynchronize(0))
    check(runtime.cudaMemPoolDestroy(pool))


def rmm_async(keep):
    previous = rmm.mr.get_current_device_resource()
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    table = plc.Table.from_arrow(pa.Table.from_pydict({"a": [1]}))
    rmm.mr.set_current_device_resource(previous)
    del table
    if keep:
        keepalive.append(mr)
    del mr


if mode == "rmm-async-destroy":
    rmm_async(keep=False)
elif mode == "rmm-async-keepalive":
    rmm_async(keep=True)
elif mode == "raw-hw-pool-destroy":
    raw_pool_alloc_destroy(runtime.cudaMemPoolCreateUsageHwDecompress)
elif mode == "raw-plain-pool-destroy":
    raw_pool_alloc_destroy(0)

tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "a.parquet")
pq.write_table(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}), path)
for _ in range(5):
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([path])
    ).build()
    plc.io.parquet.read_parquet(options)
print(f"OK {mode}")

# Copyright (c) 2022, NVIDIA CORPORATION.

import os

import psutil
import pynvml

import rmm

import cudf
from cudf.core.spill_manager import SpillManager, global_manager

_process = psutil.Process(os.getpid())


def device_mem_usage(return_total=False):
    pynvml.nvmlInit()
    idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    dev = pynvml.nvmlDeviceGetHandleByIndex(idx)
    mem = pynvml.nvmlDeviceGetMemoryInfo(dev)
    if return_total:
        return mem.total
    return mem.used


def host_mem_usage():
    return _process.memory_info().rss


rmm.reinitialize(pool_allocator=False)
manager = global_manager.reset(
    SpillManager(spill_on_demand=True)
)  # Do after RMM init
cudf.datasets.randomdata(nrows=1)  # Init cuDF


SIZE = 100_000_000
NDFS = 100
MAX_DEVICE_MEM = device_mem_usage(return_total=True)

initial_state = device_mem_usage(), host_mem_usage()
print(
    "Initial state - device: %6.3f GB, host: %6.3f GB"
    % (device_mem_usage() / 2**30, host_mem_usage() / 2**30)
)

nbytes = 0
dfs = [cudf.datasets.randomdata(nrows=SIZE)]
for i in range(NDFS):
    df = dfs[0] + i
    nbytes += df.memory_usage().sum()
    dfs.append(df)
    print(
        "[%2d] dataframes: %6.3f GB, device: %6.3f GB, host: %6.3f GB"
        % (
            i,
            nbytes / 2**30,
            device_mem_usage() / 2**30,
            host_mem_usage() / 2**30,
        )
    )
    if nbytes > MAX_DEVICE_MEM:
        break
del df

print("Spill all device memory")
evicted = 1
while evicted:
    before = device_mem_usage()
    evicted = manager.spill_device_memory()
    after = device_mem_usage()
    if evicted:
        print(
            "Spilling column: %.3f GB, device: %6.3f GB, host: %6.3f GB"
            % (
                evicted / 2**30,
                device_mem_usage() / 2**30,
                host_mem_usage() / 2**30,
            )
        )

print(
    "Finished spilling - device: %6.3f GB, host: %6.3f GB"
    % (device_mem_usage() / 2**30, host_mem_usage() / 2**30)
)

print("Access spilled dataframes")
first_df = dfs[0]
for i, df in enumerate(dfs[1:]):
    first_df += df
    print(
        "[%2d] dataframe access, device: %6.3f GB, host: %6.3f GB"
        % (
            i,
            device_mem_usage() / 2**30,
            host_mem_usage() / 2**30,
        )
    )
del first_df
del df

device_limit = MAX_DEVICE_MEM // 2
manager.spill_to_device_limit(device_limit=device_limit)
print(
    "Evict to device-limit %d GB - device: %6.3f GB, host: %6.3f GB"
    % (
        device_limit / 2**30,
        device_mem_usage() / 2**30,
        host_mem_usage() / 2**30,
    )
)

del dfs
print(
    "Deleting dataframes - device: %6.3f GB, host: %6.3f GB"
    % (device_mem_usage() / 2**30, host_mem_usage() / 2**30)
)

delta = (
    device_mem_usage() - initial_state[0],
    host_mem_usage() - initial_state[1],
)
print(
    "Initial/end state delta - device: %2.6f GB, host: %2.6f GB"
    % (delta[0] / 2**30, delta[1] / 2**30)
)

assert len(tuple(manager.base_buffers())) == 0

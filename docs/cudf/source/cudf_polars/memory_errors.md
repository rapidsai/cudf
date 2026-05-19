(cudf-polars-memory-errors)=
# Troubleshooting Memory Issues

This page covers the most common causes of out-of-memory (OOM) errors and slow spilling
when using the GPU streaming engine, and how to address them.

## Out-of-memory errors

### Lower the target partition size

The GPU streaming engine decomposes input data into a stream of chunks that can be
operated on in GPU memory independently. `target_partition_size` sets the ideal size
of each chunk in bytes. Reducing this size makes it easier for the engine to stay within
its memory budget, since each chunk is a smaller unit of work to spill and unspill. The
trade-off is that smaller chunks may reduce compute efficiency.

```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={"target_partition_size": 64_000_000},  # 64 MB
)
```

Or via environment variable:

```
CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE=64000000
```

The default is the lesser of 1.5 GB and 2.5% of the smallest GPU in the system.
For pathological workloads with heavy shuffling or skewed data distributions, a
much smaller value may be necessary.

### Lower the spill threshold

`RAPIDSMPF_SPILL_DEVICE_LIMIT` controls when spilling to host memory begins.
Setting it lower causes spilling to start earlier, leaving more room for peak
allocations:

```
RAPIDSMPF_SPILL_DEVICE_LIMIT=50%
```

The default is `80%`. You can also set this via executor options:

```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={"spill_device_limit": "50%"},
)
```

## Slow spilling

If spilling is occurring but performance is poor, enabling pinned (page-locked) host
memory can significantly speed up GPU ↔ host data transfers:

```
RAPIDSMPF_PINNED_MEMORY=true
RAPIDSMPF_PINNED_INITIAL_POOL_SIZE=32GB  # pre-allocate the full pool up front
```

```{note}
Registering page-locked memory with the driver has a significant up-front cost.
Once registration is complete, GPU ↔ host transfers will be much faster, but the
cost is only worthwhile if spilling is a clear bottleneck in your workflow, or
if you are executing many queries in the same session (so the cost can be amortized).
```

## All RapidsMPF memory options

For the full list of spill and memory configuration options see the
[RapidsMPF configuration reference](https://docs.rapids.ai/api/rapidsmpf/stable/configuration/#general).

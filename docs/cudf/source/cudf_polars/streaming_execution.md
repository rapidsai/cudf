# Experimental streaming and multi-GPU options

As well as in-memory execution, obtained by selecting `engine="gpu"`
when `collect`ing a query, the GPU engine supports streaming execution
that partitions the data in the query into chunks. To select streaming
execution, we need to pass an appropriately configured `GPUEngine`
object into `collect`.

The streaming executors work best when the inputs to your query come
from parquet files. That is, start with `scan_parquet`, not existing
Polars `DataFrame`s or CSV files.

````{note}
Streaming execution is in an experimental state and not all queries
will run in streaming mode.
````

## Single GPU streaming

The simplest case, requiring no additional dependencies, is the
`synchronous` scheduler. An appropriate engine is:

```python
engine = pl.GPUEngine(executor="streaming")
```

This uses the default synchronous *scheduler* and is equivalent to
`pl.GPUEngine(executor="streaming", executor_options={"scheduler": "synchronous"})`.

When executed with this engine, any parquet inputs are split into
"partitions" that are streamed through the query graph. We try to
pick a good default for the typical partition size (based on the
amount of GPU memory available), however, it might not be optimal. You
can configure the execution by providing more options to the executor.
For example, to split input parquet files into 125 MB chunks:

```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "target_partition_size": 125_000_000  # 125 MB
    }
)
```

Use the executor option `max_rows_per_partition` to control how in-memory
``DataFrame`` inputs are split into multiple partitions.

You may find, at the cost of higher memory footprint, that a larger value gives
better performance.

````{note}
If part of a query does not run in streaming mode, but _does_ run
using the in-memory GPU engine, then we automatically concatenate the
inputs for that operation into a single partition, and effectively
fall back to the in-memory engine.

The `fallback_mode` option can be used to raise an exception when
this fallback occurs or silence the warning instead:


    engine = pl.GPUEngine(
        executor="streaming",
        executor_options={
            "fallback_mode": "raise",
        }
    )
````

## Multi GPU streaming

Streaming utilising multiple GPUs simultaneously is supported by
setting the `"scheduler"` to `"distributed"`:
```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={"scheduler": "distributed"},
)
```

Unlike the single GPU executor, this does require a number of
additional dependencies. We currently require
[Dask](https://www.dask.org/) and
[Dask-CUDA](https://docs.rapids.ai/api/dask-cuda/nightly/) to be
installed. In addition, we recommend that
[ucxx](https://github.com/rapidsai/ucxx) and
[rapidsmpf](https://github.com/rapidsai/rapidsmpf) are installed to
take advantage of any high-performance networking.

To quickly install all of these dependencies into a conda environment,
you can run:

```
conda install -c rapidsai -c conda-forge \
    cudf-polars rapidsmpf dask-cuda ucxx
```


````{note}
Identically to the single-GPU streaming case, if part of a query does
not support execution with multiple partitions, but is supported by
the in-memory GPU engine, we concatenate the inputs and execute using
a single partition.
````

The multi-GPU engine uses the currently active Dask client to carry
out the partitioned execution, so for multi-GPU we would use something
like

```python
from dask_cuda import LocalCUDACluster

...

client = LocalCUDACluster(...).get_client()

q = ...
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={"scheduler": "distributed"},
)
result = q.collect(engine=engine)
```

````{warning}
If you request a `"distributed"` scheduler but do not have a cluster
deployed, `collect`ing the query will fail.
````

## Get Started

The experimental streaming GPU executor is now available. For a quick
walkthrough of a multi-GPU example workflow and performance on a real dataset,
check out the [multi-GPU Polars demo](https://github.com/rapidsai-community/showcase/blob/main/accelerated_data_processing_examples/multi_gpu_polars_demo.ipynb).

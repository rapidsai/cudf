# <div align="left"><img src="../../img/rapids_logo.png" width="90px"/>&nbsp;Dask cuDF - A GPU Backend for Dask DataFrame</div>

Dask cuDF (a.k.a. dask-cudf or `dask_cudf`) is an extension library for [Dask DataFrame](https://docs.dask.org/en/stable/dataframe.html). When installed, Dask cuDF is automatically registered as the `"cudf"` [dataframe backend](https://docs.dask.org/en/stable/how-to/selecting-the-collection-backend.html) for Dask DataFrame.

## Using Dask cuDF

### The Dask DataFrame API (Recommended)

Simply set the `"dataframe.backend"` [configuration](https://docs.dask.org/en/stable/configuration.html) to `"cudf"` in Dask, and the public Dask DataFrame API will leverage `cudf` automatically:

```python
import dask
dask.config.set({"dataframe.backend": "cudf"})

import dask.dataframe as dd
# This gives us a cuDF-backed dataframe
df = dd.read_parquet("data.parquet", ...)
```

> [!IMPORTANT]
> The `"dataframe.backend"` configuration will only be used for collection creation when the following APIs are used: `read_parquet`, `read_json`, `read_csv`, `read_orc`, `read_hdf`, and `from_dict`. For example, if `from_map`, `from_pandas`, `from_delayed`, or `from_array` are used, the backend of the new collection will depend on the input to the function:

```python
import pandas as pd
import cudf

# This gives us a Pandas-backed dataframe
dd.from_pandas(pd.DataFrame({"a": range(10)}))

# This gives us a cuDF-backed dataframe
dd.from_pandas(cudf.DataFrame({"a": range(10)}))
```

A cuDF-backed DataFrame collection can be moved to the `"pandas"` backend:

```python
df = df.to_backend("pandas")
```

Similarly, a Pandas-backed DataFrame collection can be moved to the `"cudf"` backend:

```python
df = df.to_backend("cudf")
```

### The Explicit Dask cuDF API

In addition to providing the `"cudf"` backend for Dask DataFrame, Dask cuDF also provides an explicit `dask_cudf` API:

```python
import dask_cudf

# This always gives us a cuDF-backed dataframe
df = dask_cudf.read_parquet("data.parquet", ...)
```

This API is used implicitly by the Dask DataFrame API when the `"cudf"` backend is enabled. Therefore, using it directly will not provide any performance benefit over the CPU/GPU-portable `dask.dataframe` API. Also, using some parts of the explicit API are incompatible with automatic query planning (see the next section).

See the [Dask cuDF's API documentation](https://docs.rapids.ai/api/dask-cudf/stable/) for further information.

## Query Planning

Dask cuDF now provides automatic query planning by default (RAPIDS 24.06+). As long as the `"dataframe.query-planning"` configuration is set to `True` (the default) when `dask.dataframe` is first imported, [Dask Expressions](https://github.com/dask/dask-expr) will be used under the hood.

For example, the following code will automatically benefit from predicate pushdown when the result is computed.

```python
df = dd.read_parquet("/my/parquet/dataset/")
result = df.sort_values('B')['A']
```

Unoptimized expression graph (`df.pprint()`):
```
Projection: columns='A'
  SortValues: by=['B'] shuffle_method='tasks' options={}
    ReadParquetFSSpec: path='/my/parquet/dataset/' ...
```

Simplified expression graph (`df.simplify().pprint()`):
```
Projection: columns='A'
  SortValues: by=['B'] shuffle_method='tasks' options={}
    ReadParquetFSSpec: path='/my/parquet/dataset/' columns=['A', 'B'] ...
```

> [!NOTE]
> Dask will automatically simplify the expression graph (within `optimize`) when the result is converted to a task graph (via `compute` or `persist`). You do not need to call `simplify` yourself.


## Using Multiple GPUs and Multiple Nodes

Whenever possible, Dask cuDF (i.e. Dask DataFrame) will automatically try to partition your data into small-enough tasks to fit comfortably in the memory of a single GPU. This means the necessary compute tasks needed to compute a query can often be streamed to a single GPU process for out-of-core computing. This also means that the compute tasks can be executed in parallel over a multi-GPU cluster.

> [!IMPORTANT]
> Neither Dask cuDF nor Dask DataFrame provide support for multi-GPU or multi-node execution on their own. You must deploy a distributed cluster (ideally with [Dask CUDA](https://docs.rapids.ai/api/dask-cuda/stable/)) to leverage multiple GPUs.

In order to execute your Dask workflow on multiple GPUs, you will typically need to use [Dask CUDA](https://docs.rapids.ai/api/dask-cuda/stable/) to deploy distributed Dask cluster, and [Distributed](https://distributed.dask.org/en/stable/client.html) to define a `client` object. For example:

```python

from dask_cuda import LocalCUDACluster
from distributed import Client

client = Client(
    LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0,1",  # Use two workers (on devices 0 and 1)
        rmm_pool_size=0.9,  # Use 90% of GPU memory as a pool for faster allocations
        enable_cudf_spill=True,  # Improve device memory stability
        local_directory="/fast/scratch/",  # Use fast local storage for spilling
    )
)

df = dd.read_parquet("/my/parquet/dataset/")
agg = df.groupby('B').sum()
agg.compute()  # This will use the cluster defined above
```

> [!NOTE]
> This example uses `compute` to materialize a concrete `cudf.DataFrame` object in local memory. Never call `compute` on a large collection that cannot fit comfortably in the memory of a single GPU! See Dask's [documentation on managing computation](https://distributed.dask.org/en/stable/manage-computation.html) for more details.

Please see the [Dask CUDA](https://docs.rapids.ai/api/dask-cuda/stable/) documentation for more information about deploying GPU-aware clusters (including [best practices](https://docs.rapids.ai/api/dask-cuda/stable/examples/best-practices/)).

## Install

See the [RAPIDS install page](https://docs.rapids.ai/install) for the most up-to-date information and commands for installing Dask cuDF and other RAPIDS packages.

## Resources

- [Dask cuDF API documentation](https://docs.rapids.ai/api/dask-cudf/stable/)
- [cuDF API documentation](https://docs.rapids.ai/api/cudf/stable/)
- [10 Minutes to cuDF and Dask cuDF](https://docs.rapids.ai/api/cudf/stable/user_guide/10min/)
- [Dask CUDA documentation](https://docs.rapids.ai/api/dask-cuda/stable/)
- [Deployment](https://docs.rapids.ai/deployment/stable/)
- [RAPIDS Community](https://rapids.ai/learn-more/#get-involved): Get help, contribute, and collaborate.

(cudf-polars-api)=
# API Reference

For the most part, the public API of `cudf-polars` is the Polars API itself. This page
documents the additional classes and functions that `cudf-polars` exposes for the streaming
multi-GPU engines.

## Streaming engines

```{eval-rst}
.. autoclass:: cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine
   :members: from_options, gather_cluster_info, shutdown, nranks
   :show-inheritance:

.. autoclass:: cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine
   :members: from_options, gather_cluster_info, shutdown, nranks
   :show-inheritance:

.. autoclass:: cudf_polars.experimental.rapidsmpf.frontend.spmd.SPMDEngine
   :members: from_options, gather_cluster_info, shutdown, nranks, rank, comm, context
   :show-inheritance:
```

The three engines share a common base class:

```{eval-rst}
.. autoclass:: cudf_polars.experimental.rapidsmpf.frontend.core.StreamingEngine
   :members: gather_cluster_info, shutdown, nranks
   :show-inheritance:

.. autoclass:: cudf_polars.experimental.rapidsmpf.frontend.core.ClusterInfo
   :members:
```

## Configuration

```{eval-rst}
.. autoclass:: cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions
   :members: from_dict, to_dict, to_rapidsmpf_options, to_executor_options, to_engine_options

.. autodata:: cudf_polars.experimental.rapidsmpf.frontend.options.UNSPECIFIED

.. autoclass:: cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.HardwareBindingPolicy
   :members:
```

## SPMD helpers

```{eval-rst}
.. autofunction:: cudf_polars.experimental.rapidsmpf.frontend.spmd.allgather_polars_dataframe

.. autofunction:: cudf_polars.experimental.rapidsmpf.collectives.common.reserve_op_id
```

## Internal configuration objects

These dataclasses back the `engine_options` surfaced by `pl.GPUEngine` and `StreamingOptions`.
Most users interact with them through `StreamingOptions` fields rather than directly.

```{eval-rst}
.. automodule:: cudf_polars.utils.config
   :members:
      DynamicPlanningOptions,
      MemoryResourceConfig,
      ParquetOptions,
```

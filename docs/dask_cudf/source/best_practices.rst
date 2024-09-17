.. _best-practices:

Dask cuDF Best Practices
========================

This page outlines several important guidelines for using Dask cuDF
effectively.

.. note::
  Since Dask cuDF is a backend extension for
  `Dask DataFrame <https://docs.dask.org/en/stable/dataframe.html>`__,
  many of the details discussed in the `Dask DataFrames Best Practices
  <https://docs.dask.org/en/stable/dataframe-best-practices.html>`__
  documentation also apply to Dask cuDF.


Deployment and Configuration
----------------------------

Use Dask DataFrame Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although Dask cuDF provides a public ``dask_cudf`` Python module, we
strongly recommended that you use the CPU/GPU portable ``dask.dataframe``
API instead. Simply use the `Dask configuration <dask:configuration>`__
system to set the ``"dataframe.backend"`` option to ``"cudf"``, and
the ``dask_cudf`` module will be imported and used implicitly.

Use Dask-CUDA
~~~~~~~~~~~~~

In order to execute a Dask workflow on multiple GPUs, a Dask "cluster" must
be deployed with `Dask-CUDA <https://docs.rapids.ai/api/dask-cuda/stable/>`__
and/or `Dask.distributed <https://distributed.dask.org/en/stable/>`__.

When running on a single machine, the `LocalCUDACluster <https://docs.rapids.ai/api/dask-cuda/stable/api/#dask_cuda.LocalCUDACluster>`__
convenience function is strongly recommended. No matter how many GPUs are
available on the machine (even one!), using Dask-CUDA has important advantages
over default (threaded) execution::

* Dask-CUDA makes it easy to pin workers to specific devices.
* Dask-CUDA makes it easy to configure memory-spilling options.
* The distributed scheduler collects useful diagnostic information that can be viewed on a dashboard in real time.

Please see `Dask-CUDA's API <https://docs.rapids.ai/api/dask-cuda/stable/>`__
and `Best Practices <https://docs.rapids.ai/api/dask-cuda/stable/examples/best-practices/>`__
documentation for detailed information.

.. note::
  When running on cloud infrastructure or HPC systems, it is usually best to
  leverage system-specific deployment libraries like `Dask Operator
  <https://docs.dask.org/en/latest/deploying-kubernetes.html>`__ and `Dask-Jobqueue
  <https://jobqueue.dask.org/en/latest/>`__.

  Please see `RAPIDS-deployment documentation <https://docs.rapids.ai/deployment/stable/>`__
  for further details and examples.

Enable cuDF Spilling
~~~~~~~~~~~~~~~~~~~~

When using Dask cuDF for classic ETL workloads, it is usually best
to enable `native spilling support in cuDF
<https://docs.rapids.ai/api/cudf/stable/developer_guide/library_design/#spilling-to-host-memory>`__.
When using :func:`LocalCUDACluster`, this is easily accomplished by
setting ``enable_cudf_spill=True``.

When a Dask cuDF workflow includes conversion between DataFrame and Array
representations, native cuDF spilling may be insufficient. For these cases,
JIT unspilling is likely to produce better protection from out-of-memory
(OOM) errors. Please see `Dask-CUDA's spilling documentation
<https://docs.rapids.ai/api/dask-cuda/24.10/spilling/>`__ for further details
and guidance.


Reading Data
------------

Tune the partition size
~~~~~~~~~~~~~~~~~~~~~~~

The ideal partition size is typically between 2-10% of the memory capacity
of a single GPU. Increasing the partition size will typically reduce the
number of tasks in your workflow and improve the GPU utilization for each
task. However, if the partitions are too large, the risk of OOM errors can
become significant.

The best way to tune the partition size is to begin with appropriate sized
partitions when the DataFrame collection is first created by a function
like :func:`read_parquet`, :func:`read_csv`, or :func:`from_map`. For
example, both :func:`read_parquet` and :func:`read_csv` expose a
``blocksize`` argument for adjusting the maximum partition size.

If the partition size cannot be tuned effectively at creation time, the
`repartition <https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.repartition.html>`__
method can be used as a last resort.


Use Parquet files
~~~~~~~~~~~~~~~~~

`Parquet <https://parquet.apache.org/docs/file-format/>`__ is the recommended
file format for Dask cuDF. It provides efficient columnar storage and enables
Dask to perform valuable query optimizations like column projection and
predicate pushdown.

The most important arguments to :func:`read_parquet` are ``blocksize`` and
``aggregate_files``::

``blocksize``: Use this argument to specify the maximum partition size.
The default is `"256 MiB"`, but larger values are usually more performant
(e.g. `1 GiB` is usually safe). Dask will use the ``blocksize`` value to map
a discrete number of Parquet row-groups (or files) to each output partition.
This mapping will only account for the uncompressed storage size of each
row group, which is usually smaller than the correspondng ``cudf.DataFrame``.

``aggregate_files``: Use this argument to specify whether Dask is allowed
to map multiple files to the same DataFrame partition. The default is ``False``,
but ``aggregate_files=True`` is usually more performant when the dataset
contains many files that are smaller than half of ``blocksize``.

.. note::
  Metadata collection can be extremely slow when reading from remote
  storage (e.g. S3 and GCS). When reading many remote files that all
  correspond to a reasonable partition size, it's usually best to set
  `blocksize=None` and `aggregate_files=False`. In most cases, these
  settings allow Dask to skip the metadata-collection stage altogether.


Use :func:`from_map`
~~~~~~~~~~~~~~~~~~~~

To implement custom DataFrame-creation logic that is not covered by
existing APIs (like :func:`read_parquet`),  use :func:`dask.dataframe.from_map`
whenever possible. The :func:`from_map` API has several advantages
over :func:`from_delayed`::

* It allows proper lazy execution of your custom logic
* It enables column projection (as long as the mapped function supports a ``columns`` key-word argument)

See the `from_map API documentation <https://docs.dask.org/en/stable/generated/dask_expr.from_map.html#dask_expr.from_map>`__
for more details.


Sorting, Joining and Grouping
-----------------------------

Sorting, joining and grouping operations all have the potential to
require the global shuffling of data between distinct partitions.
When the initial data fits comfortably in global GPU memory, these
"all-to-all" operations are typically bound by worker-to-worker
communication. When the data is larger than global GPU memory, the
bottleneck is typically device-to-host memory spilling.

Although every workflow is different, the following guidelines
are often recommended::

* Use a distributed cluster with Dask-CUDA workers
* Use native cuDF spilling whenever possible
* Avoid shuffling whenever possible
  * Use ``split_out=1`` for low-cardinality groupby aggregations
  * Use ``broadcast=True`` for joins when at least one collection comprises a small number of partitions (e.g. ``>=5``)
* `Use UCX <https://docs.rapids.ai/api/dask-cuda/nightly/examples/ucx/>`__ if communication is a bottleneck


User-defined functions
----------------------

Most real-world Dask DataFrame workflows use `map_partitions
<https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html>`__
to map user-defined functions across every partition of the underlying data.
This API is a fantastic way to apply custom operations in an intuitive and
scalable way. With that said, the :func:`map_partitions` method will produce
in an opaque DataFrame expression that blocks the query-planning `optimizer
<https://docs.dask.org/en/stable/dataframe-optimizer.html>`__ from performing
useful optimizations (like projection and filter pushdown).

Since column-projection pushdown is often the most important optimization,
you can mitigate the loss of these optimizations by explicitly selecting
the necessary columns both before and after calling :func:`map_partitions`.
Adding explicit filter operations may further mitigate the loss of filter
pushdown.

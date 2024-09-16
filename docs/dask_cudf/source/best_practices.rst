.. _best-practices:

Dask cuDF Best Practices
========================

Deployment
----------

Use Dask-CUDA
~~~~~~~~~~~~~

In order to execute a Dask workflow on multiple GPUs, a Dask "cluster" must
be deployed with Dask Distributed and/or Dask-CUDA.

When running on a single machine, :func:`dask_cuda.LocalCUDACluster` is strongly
recommended. This is still true when there is only a single GPU on the machine.
No matter the device count, using Dask-CUDA has several advantages over default
(threaded) execution::

* Dask-CUDA makes it easy to pin workers to specific devices.
* Dask-CUDA makes it easy to configure memory-spilling option.
* The distributed scheduler collects useful diagnostic information that can be viewed on a dashboard in real time.

Please see `Dask-CUDA's API <https://docs.rapids.ai/api/dask-cuda/stable/>`_
and `Best Practices <https://docs.rapids.ai/api/dask-cuda/stable/examples/best-practices/>`_
documentation for detailed information.

.. note::
  When running on cloud infrastructure or HPC systems, it is often best to
  leverage system-specific deployment libraries like Dask Operator and Dask
  Jobqueue.

  Please see `RAPIDS-deployment documentation <https://docs.rapids.ai/deployment/stable/>`_
  for further details and examples.

Enable cuDF Spilling
~~~~~~~~~~~~~~~~~~~~

When using Dask cuDF for classic ETL workloads, it is usually best
to enable `native spilling support in cuDF
<https://docs.rapids.ai/api/cudf/stable/developer_guide/library_design/#spilling-to-host-memory>`_.
When using :func:`LocalCUDACluster`, this is easily accomplished by
setting ``enable_cudf_spill=True``.

When a Dask cuDF workflow includes conversion between DataFrame and Array
representations, native cuDF spilling may be insufficient. For these cases,
JIT unspilling is likely to produce better protection from out-of-memory
(OOM) errors.

Please see `Dask-CUDA's spilling documentation
<https://docs.rapids.ai/api/dask-cuda/24.10/spilling/>`_ for more details.


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
partitions when the DataFrame collection is first created with a function
like :func:`read_parquet`, :func:`read_csv`, or :func:`from_map`. Both
:func:`read_parquet` and :func:`read_csv` expose a ``blocksize`` argument
for adjusting the maximum partition size.

If the partition size cannot be tuned effectively at creation time, the
:func:`DataFrame.repartition` method can be used as a last resort.


Use Parquet files
~~~~~~~~~~~~~~~~~

`Parquet <https://parquet.apache.org/docs/file-format/>`_ is the recommended
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

See the `from_map API documentation <https://docs.dask.org/en/stable/generated/dask_expr.from_map.html#dask_expr.from_map>`_
for more details.


Sorting, Joining and Grouping
-----------------------------

* Make sure spilling is enabled
* Avoid shuffling whenever possible (e.g. use broadcast joins)
  * Use ``split_out=1`` for low-cardinality groupby aggregations
  * Use ``broadcast=True`` when at least one collection comprises a small number of partitions (e.g. ``>=5``).
* Use UCX if communication is a bottleneck

User-defined functions
----------------------

* Use :func:`map_partitions`
* Select the necessary columns before/after :func:`map_partitions`

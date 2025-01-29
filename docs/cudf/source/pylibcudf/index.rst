pylibcudf documentation
=======================

pylibcuDF is a lightweight Cython interface to libcuDF that provides near-zero overhead for GPU-accelerated data processing in Python.
It aims to match native C++ performance of libcuDF while integrating seamlessly with community protocols like ``__cuda_array_interface__``, and common libraries such as CuPy and Numba.
Both our zero-code pandas accelerator (``cudf.pandas``) and our polars GPU execution engine (``cudf.polars``) are built on top of pylibcuDF.

Ex: Reading data from a parquet file

pylibcuDF:

.. code-block:: python

   import pylibcudf as plc

   source = plc.io.SourceInfo(["dataset.parquet"])
   options = plc.io.parquet.ParquetReaderOptions.builder(source)
   table = plc.io.parquet.read_parquet(options)

libcuDF:

.. code-block:: cpp

   #include <cudf>

   auto source  = cudf::io::source_info("dataset.parquet");
   auto options = cudf::io::csv_reader_options::builder(source);
   auto table  = cudf::io::read_csv(options);

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   api_docs/index.rst
   developer_docs

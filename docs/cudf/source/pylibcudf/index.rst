pylibcudf documentation
=======================

pylibcuDF provides low-level python bindings for the libcuDF C++ library. As of 25.02, it covers most of the libcuDF API.

Ex: Reading from a parquet file

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

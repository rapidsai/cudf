cuDF-based GPU backend for Polars [Open Beta]
=============================================

cuDF supports an in-memory, GPU-accelerated execution engine for Python users of the Polars Lazy API.
The engine supports most of the core expressions and data types as well as a growing set of more advanced dataframe manipulations
and data file formats. When using the GPU engine, Polars will convert expressions into an optimized query plan and determine
whether the plan is supported on the GPU. If it is not, the execution will transparently fall back to the standard Polars engine
and run on the CPU.

Benchmark
---------
We reproduced the `Polars Decision Support (PDS) <https://github.com/pola-rs/polars-benchmark>`__ benchmark to compare Polars GPU engine with the default CPU settings across several dataset sizes. Here are the results:

.. figure:: ../_static/pds_benchmark_polars.png
   :width: 600px



You can see up to a 13x speed-up on the top-performing, compute-heavy PDS queries involving complex aggregation and join operations on the 80 GB dataset.


.. figure:: ../_static/compute_heavy_queries_polars.png
   :width: 1000px

You can reproduce the results by visiting the `Polars Decision Support (PDS) Github repository <https://github.com/pola-rs/polars-benchmark>`__.

Learn More
----------

The GPU backend for Polars is now available in Open Beta and the engine is undergoing rapid development. To learn more, visit the `GPU Support page <https://docs.pola.rs/user-guide/gpu-support/>`__ on the Polars website.

Launch on Google Colab
----------------------

.. figure:: ../_static/colab.png
   :width: 200px
   :target: https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/accelerated_data_processing_examples/polars_gpu_engine_demo.ipynb

   Take the cuDF backend for Polars for a test-drive in a free GPU-enabled notebook environment using your Google account by `launching on Colab <https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/accelerated_data_processing_examples/polars_gpu_engine_demo.ipynb>`__.

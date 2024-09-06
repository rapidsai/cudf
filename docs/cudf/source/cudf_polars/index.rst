cuDF-based GPU backend for Polars [Open Beta]
================================

cuDF supports an in-memory, GPU-accelerated execution engine for Python users of the Polars Lazy API. 
The engine supports most of the core expressions and data types as well as a growing set of more advanced dataframe manipulations 
and data file formats. When using the GPU engine, Polars will convert expressions into an optimized query plan and determine 
whether the plan is supported on the GPU. If it is not, the execution will transparently fall back to the standard Polars engine 
and run on the CPU. 

<TO-DO: Benchmarks>

Learn More
----------------------

The GPU backend for Polars is now available in Open Beta and the engine is undergoing rapid development. To learn more, visit the `GPU Support page <https://docs.pola.rs/user-guide/gpu-support/>`_ on the Polars website.

Launch on Google Colab
----------------------

.. figure:: ../_static/colab.png
   :width: 200px
   :target: https://nvda.ws/rapids-cudf

   Take the cuDF backend for Polars for a test-drive in a free GPU-enabled notebook environment using your Google account by `launching on Colab <TBD>`_  

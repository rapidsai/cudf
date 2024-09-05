cuDF-based GPU backend for Polars
================================

cuDF supports an in-memory, GPU-accelerated execution engine for Python users of the Polars Lazy API. 
The engine supports most of the core expressions and data types as well as a growing set of more advanced dataframe manipulations 
and data file formats. When using the GPU engine, Polars will convert expressions into an optimized query plan and determine 
whether the plan is supported on the GPU. If it is not, the execution will transparently fall back to the standard Polars engine 
and run on the CPU. You can install the GPU backend for Polars with a feature flag in your standard Python pip install command.

.. code-block:: bash

   pip install polars[gpu]

GPU-based execution can be triggered by simply running ``.collect(engine="gpu")`` instead of ``.collect()``.

.. code-block:: python

   # Import the necessary library
   import polars as pl

   # Define the data for the LazyFrame
   ldf = pl.LazyFrame({
      "a": [1.242, 1.535],
   })

   print(ldf.select(pl.col("a").round(1)).collect(engine="gpu"))


For finer control, you can pass a GPUEngine object with additional configuration parameters to the ``engine=`` parameter.

.. code-block:: python

   # Import the necessary library
   import polars as pl

   # Define the data for the LazyFrame
   ldf = pl.LazyFrame({
      "a": [1.242, 1.535],
   })

   # Configure the GPU engine with advanced settings
   gpu_engine = pl.GPUEngine(
      device=0,
      raise_on_Fail=True  # Ensure the engine fails loudly if it cannot execute on the GPU
   )

   # Execute the collection with the custom GPU engine configuration
   print(ldf.select(pl.col("a").round(1)).collect(engine=gpu_engine))

Open Beta Announcement
----------------------

   The GPU backend for Polars is now available in Open Beta and the engine is undergoing rapid development. To learn more, visit the `GPU Support page <YOUR_LINK_TO_GPU_SUPPORT_PAGE>`_ on the Polars website.

Launch on Google Colab
----------------------

.. figure:: ../_static/colab.png
    :width: 200px
    :target: <YOUR_LINK_TO_COLAB>

   Take the cuDF backend for Polars for a test-drive in a free GPU-enabled notebook environment on Google Colab.
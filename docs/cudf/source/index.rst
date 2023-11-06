Welcome to cuDF's documentation!
=================================

cuDF is a Python GPU DataFrame library (built on the `Apache Arrow
<http://arrow.apache.org/>`_ columnar memory format) for loading, joining,
aggregating, filtering, and otherwise manipulating data. cuDF also provides a
pandas-like API that will be familiar to data engineers & data scientists, so
they can use it to easily accelerate their workflows without going into
the details of CUDA programming.


``cudf.pandas`` is a 100% drop-in replacement for pandas that behaves identically on your
CPU but lets you "hit the turbo button" and run supported functions on an
NVIDIA GPU without code change.

.. image:: _static/RAPIDS-logo-purple.png
    :width: 300px
    :class: align-right


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide/index
   cudf_pandas/index
   developer_guide/index

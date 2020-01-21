Developer Documentation
=======================

Code Organization
-----------------

This shows the basic code organization.

Currently, the repo is basically flat.  All implementations are directly under
under the ``cudf/`` directory.  All tests are in ``cudf/tests/`` directory.

Here's a quick map to decide which file contains which feature:

- ``DataFrame``:
    - ``dataframe.py``
- ``Series``:
    - ``series.py``
- ``Column`` and its subclasses:
    - ``column.py``
    - ``columnops.py``
    - ``numerical.py`` for numeric columns
    - ``categorical.py`` for categorical columns
- ``Buffer``:
    - ``buffer.py``
- ``.apply()`` and simliar functions:
    - ``applyutils.py``
- ``.query()`` and similar functions:
    - ``queryutils.py``
- GPU helper functions:
    - ``cudautils.py``
- Docstring helpers:
    - ``docutils.py``
- Output formating:
    - ``formatting.py``
- Arrow:
    - ``gpuarrow.py``
- Groupby:
    - ``groupby.py``
- Dask serialization helpers:
    - ``serialize.py``
- ``Index``:
    - ``index.py``
- Operations on multiple DataFrame, Series or Indices:
    - ``multi.py``
- Other general helper functions:
    - ``utils.py``



Code that should move to libgdf
--------------------------------

Code that should be re-implemented in libgdf in CUDA-C for better
reusability and performance.

- ``cudf/cudautils.py`` contains a lot of GPU helper functions
  that are jitted by numba with ``@cuda.jit`` into CUDA kernels.
  All CUDA kernels in this file should be moved to libgdf if possible.

- Some logic in ``cudf/groupby.py`` should be move to libgdf to make
  groupby operation faster.  Some groupby aggregations are implemented with
  ``@cuda.jit`` here.


Code that cannot move to libgdf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some features requires the jit to be useful; e.g features that use
user-defined functions.  These features cannot be moved to libgdf.

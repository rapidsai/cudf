Comparison of cuDF and Pandas
=============================

cuDF is a DataFrame library that closely matches the Pandas API, but
leverages NVIDIA GPUs for performing computations for speed.  However,
there are some differences between cuDF and Pandas, both in terms API
and behavior. This page documents the similarities and differences
between cuDF and Pandas.

Data types
----------

cuDF supports many common data types supported by Pandas, including
numeric, datetime, timestamp, string, and categorical data types.  In
addition, we support special data types for decimal, list and "struct"
values.  See the section on :doc:`Data Types <data-types>` for
details.

Note that we do not support custom data types like Pandas'
``ExtensionDtype``.

Result ordering
---------------

By default, ``join`` (or ``merge``) and ``groupby`` operations in cuDF
do *not* guarantee output ordering by default.
Compare the results obtained from Pandas and cuDF below:

.. code:: python

    >>> import cupy as cp
    >>> df = cudf.DataFrame({'a': cp.random.randint(0, 1000, 1000), 'b': range(1000)})
    >>> df.groupby("a").mean().head()
             b
    a
    742  694.5
    29   840.0
    459  525.5
    442  363.0
    666    7.0
    >>> df.to_pandas().groupby("a").mean().head()
             b
    a
    2   643.75
    6    48.00
    7   631.00
    9   906.00
    10  640.00

To match Pandas behavior, you must explicitly pass ``sort=True``:

.. code:: python

    >>> df.to_pandas().groupby("a", sort=True).mean().head()
             b
    a
    2   643.75
    6    48.00
    7   631.00
    9   906.00
    10  640.00

Column names
------------

Unlike Pandas, cuDF does not support duplicate column names.
It is best to use strings for column names.

No true ``"object"`` data type
------------------------------

In Pandas and NumPy, the ``"object"`` data type is used for
collections of arbitrary Python objects.  For example, in Pandas you
can do the following:

.. code:: python
    >>> import pandas as pd
    >>> s = pd.Series(["a", 1, [1, 2, 3]])
    0            a
    1            1
    2    [1, 2, 3]
    dtype: object

For compatibilty with Pandas, cuDF reports the data type for strings
as ``"object"``, but we do *not* support storing or operating on
collections of arbitrary Python objects.

``.apply()`` function limitations
---------------------------------

The ``.apply()`` function in Pandas accecpts a user-defined function
(UDF) that can include arbitrary operations that are applied to each
value of a ``Series``, ``DataFrame``, or in the case of a groupby,
each group.  cuDF also supports ``apply()``, but it relies on Numba to
JIT compile the UDF and execute it on the GPU. This can be extremely
fast, but imposes a few limitations on what operations are allowed in
the UDF. See our :doc:`UDF docs <guide-to-udf>` for details.

How to check if a particular Pandas feature is available in cuDF?
-----------------------------------------------------------------

The best way to see if we support a particular feature is to search
our `API docs <https://docs.rapids.ai/api/cudf/stable/>`_.

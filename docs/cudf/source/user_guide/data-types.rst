Supported Data Types
====================

cuDF lets you store and operate on many different types of data on the
GPU.  Each type of data is associated with a data type (or "dtype").
cuDF supports many data types supported by NumPy and Pandas, including
numeric, datetime, timedelta, categorical and string data types.  In
addition cuDF supports special data types for decimals and "nested
types" (lists and structs).

Unlike in Pandas, all data types in cuDF are nullable.
See :doc:`Working With Missing Data </user_guide/Working-with-missing-data>`.


.. rst-class:: special-table
.. table::

    +-----------------+----------------------------+--------------------------------------------------------------+----------------------------------------------+
    | Kind of Data    | Data Type                  | Scalar                                                       | String Aliases                               |
    +=================+============================+==============================================================+==============================================+
    | Integer         |np.dtype(...)               | np.int8_, np.int16_, np.int32_, np.int64_, np.uint8_,        | ``'int8'``, ``'int16'``, ``'int32'``,        |
    |                 |                            | np.uint16_, np.uint32_, np.uint64_                           | ``'int64'``, ``'uint8'``, ``'uint16'``,      |
    |                 |                            |                                                              | ``'uint32'``, ``'uint64'``                   |
    +-----------------+----------------------------+--------------------------------------------------------------+----------------------------------------------+
    | Float           |np.dtype(...)               | np.float32_, np.float64_                                     | ``'float32'``, ``'float64'``                 |
    +-----------------+----------------------------+--------------------------------------------------------------+----------------------------------------------+
    | Strings         |np.dtype('object')          | `str <https://docs.python.org/3/library/stdtypes.html#str>`_ | ``'string'``, ``'object'``                   |
    +-----------------+----------------------------+--------------------------------------------------------------+----------------------------------------------+
    | Datetime        |np.dtype('datetime64[...]') | np.datetime64_                                               | ``'datetime64[s]'``, ``'datetime64[ms]'``,   |
    |                 |                            |                                                              | ``'datetime64[us]'``, ``'datetime64[ns]'``   |
    +-----------------+----------------------------+--------------------------------------------------------------+----------------------------------------------+
    | Timedelta       |np.dtype('timedelta64[...]')| np.timedelta64_                                              | ``'timedelta64[s]'``, ``'timedelta64[ms]'``, |
    | (duration type) |                            |                                                              | ``'timedelta64[us]'``, ``'timedelta64[ns]'`` |
    +-----------------+----------------------------+--------------------------------------------------------------+----------------------------------------------+
    | Categorical     |cudf.CategoricalDtype(...)  |(none)                                                        | ``'category'``                               |
    +-----------------+----------------------------+--------------------------------------------------------------+----------------------------------------------+
    | Boolean         |np.dtype('bool')            | np.bool_                                                     | ``'bool'``                                   |
    +-----------------+----------------------------+--------------------------------------------------------------+----------------------------------------------+
    | Decimal         |cudf.Decimal32Dtype(...),   |(none)                                                        |(none)                                        |
    |                 |cudf.Decimal64Dtype(...),   |                                                              |                                              |
    |                 |cudf.Decimal128Dtype(...)   |                                                              |                                              |
    +-----------------+----------------------------+--------------------------------------------------------------+----------------------------------------------+
    | Lists           |cudf.ListDtype(...)         | list                                                         |(none)                                        |
    +-----------------+----------------------------+--------------------------------------------------------------+----------------------------------------------+
    | Structs         |cudf.StructDtype(...)       | dict                                                         |(none)                                        |
    +-----------------+----------------------------+--------------------------------------------------------------+----------------------------------------------+


A note on strings
-----------------

The data type associated with string data in cuDF is ``"object"``.

.. code:: python
    >>> import cudf
    >>> s = cudf.Series(["abc", "def", "ghi"])
    >>> s.dtype
    dtype("object")

This is for compatibility with Pandas, but it can be misleading. In
both NumPy and Pandas, ``"object"`` is the data type associated data
composed of arbitrary Python objects (not just strings).  However,
cuDF does not support storing arbitrary Python objects.


.. _np.int8:
.. _np.int16:
.. _np.int32:
.. _np.int64:
.. _np.uint8:
.. _np.uint16:
.. _np.uint32:
.. _np.uint64:
.. _np.float32:
.. _np.float64:
.. _np.bool: https://numpy.org/doc/stable/user/basics.types.html
.. _np.datetime64: https://numpy.org/doc/stable/reference/arrays.datetime.html#basic-datetimes
.. _np.timedelta64: https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-and-timedelta-arithmetic

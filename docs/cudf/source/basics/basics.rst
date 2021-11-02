Basics
======


Supported Dtypes
----------------

cuDF uses dtypes for Series or individual columns of a DataFrame. cuDF uses NumPy dtypes, NumPy provides support for ``float``, ``int``, ``bool``,
``'timedelta64[s]'``, ``'timedelta64[ms]'``, ``'timedelta64[us]'``, ``'timedelta64[ns]'``, ``'datetime64[s]'``, ``'datetime64[ms]'``,
``'datetime64[us]'``, ``'datetime64[ns]'`` (note that NumPy does not support timezone-aware datetimes).


The following table lists all of cudf types. For methods requiring dtype arguments, strings can be specified as indicated. See the respective documentation sections for more on each type.

.. rst-class:: special-table
.. table::

    +------------------------+------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Kind of Data           | Data Type        | Scalar                                                                              | String Aliases                              |
    +========================+==================+=====================================================================================+=============================================+
    | Integer                |                  | np.int8_, np.int16_, np.int32_, np.int64_, np.uint8_, np.uint16_,                   | ``'int8'``, ``'int16'``, ``'int32'``,       |
    |                        |                  | np.uint32_, np.uint64_                                                              | ``'int64'``, ``'uint8'``, ``'uint16'``,     |
    |                        |                  |                                                                                     | ``'uint32'``, ``'uint64'``                  |
    +------------------------+------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Float                  |                  | np.float32_, np.float64_                                                            | ``'float32'``, ``'float64'``                |
    +------------------------+------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Strings                |                  | `str <https://docs.python.org/3/library/stdtypes.html#str>`_                        | ``'string'``, ``'object'``                  |
    +------------------------+------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Datetime               |                  | np.datetime64_                                                                      | ``'datetime64[s]'``, ``'datetime64[ms]'``,  |
    |                        |                  |                                                                                     | ``'datetime64[us]'``, ``'datetime64[ns]'``  |
    +------------------------+------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Timedelta              |                  | np.timedelta64_                                                                     | ``'timedelta64[s]'``, ``'timedelta64[ms]'``,|
    | (duration type)        |                  |                                                                                     | ``'timedelta64[us]'``, ``'timedelta64[ns]'``|
    +------------------------+------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Categorical            | CategoricalDtype | (none)                                                                              | ``'category'``                              |
    +------------------------+------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Boolean                |                  | np.bool_                                                                            | ``'bool'``                                  |
    +------------------------+------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Decimal                | Decimal32Dtype,  | (none)                                                                              | (none)                                      |
    |                        | Decimal64Dtype   |                                                                                     |                                             |
    +------------------------+------------------+-------------------------------------------------------------------------------------+---------------------------------------------+

**Note: All dtypes above are Nullable**

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

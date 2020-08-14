Basics
======


Supported Dtypes
----------------

cuDF uses dtypes for Series or individual columns of a DataFrame. cuDF uses NumPy dtypes, NumPy provides support for ``float``, ``int``, ``bool``,
``'timedelta64[s]'``, ``'timedelta64[ms]'``, ``'timedelta64[us]'``, ``'timedelta64[ns]'``, ``'datetime64[s]'``, ``'datetime64[ms]'``,
``'datetime64[us]'``, ``'datetime64[ns]'`` (note that NumPy does not support timezone-aware datetimes).


The following table lists all of cudf types. For methods requiring dtype arguments, strings can be specified as indicated. See the respective documentation sections for more on each type.


+------------------------+------------+-------------------------------------------------------------------------------------+---------------------------------------------+
| Kind of Data           | Data Type  | Scalar                                                                              | String Aliases                              |
| (header rows optional) |            |                                                                                     |                                             |
+========================+============+=====================================================================================+=============================================+
| Integer                |            | `int <https://docs.python.org/3/library/functions.html#int>`_                       | ``'int8'``, ``'int16'``, ``'int32'``,       |
|                        |            |                                                                                     | ``'int64'``, ``'uint8'``, ``'uint16'``,     |
|                        |            |                                                                                     | ``'uint32'``, ``'uint64'``                  |
+------------------------+------------+-------------------------------------------------------------------------------------+---------------------------------------------+
| Float                  |            | `float <https://docs.python.org/3/library/functions.html#float>`_                   | ``'float32'``, ``'float64'``                |
+------------------------+------------+-------------------------------------------------------------------------------------+---------------------------------------------+
| Strings                |            | `str <https://docs.python.org/3/library/stdtypes.html#str>`_                        | ``'string'``, ``'object'``                  |
+------------------------+------------+-------------------------------------------------------------------------------------+---------------------------------------------+
| Datetime               |            | `numpy.datetime64 <https://numpy.org/doc/stable/reference/arrays.datetime.html>`_   | ``'datetime64[s]'``, ``'datetime64[ms]'``,  |
|                        |            |                                                                                     | ``'datetime64[us]'``, ``'datetime64[ns]'``  |
+------------------------+------------+-------------------------------------------------------------------------------------+---------------------------------------------+
| Timedelta              |            | `numpy.timedelta64 <https://numpy.org/doc/stable/reference/arrays.datetime.html>`_  | ``'timedelta64[s]'``, ``'timedelta64[ms]'``,|
| (duration type)        |            |                                                                                     | ``'timedelta64[us]'``, ``'timedelta64[ns]'``|
+------------------------+------------+-------------------------------------------------------------------------------------+---------------------------------------------+
| Categorical            |            | (none)                                                                              | ``'category'``                              |
+------------------------+------------+-------------------------------------------------------------------------------------+---------------------------------------------+
| Boolean                |            | `bool <https://docs.python.org/3/library/functions.html#bool>`_                     | ``'bool'``                                  |
+------------------------+------------+-------------------------------------------------------------------------------------+---------------------------------------------+
| List                   |  ListDtype | `list <https://docs.python.org/3/tutorial/datastructures.html#more-on-lists>`_      | ``'list'``                                  |
+------------------------+------------+-------------------------------------------------------------------------------------+---------------------------------------------+

**Note: All dtypes above are Nullable**
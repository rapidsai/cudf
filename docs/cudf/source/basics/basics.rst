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

    +------------------------+------------------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Kind of Data           | Data Type                    | Scalar                                                                              | String Aliases                              |
    +========================+==============================+=====================================================================================+=============================================+
    | Integer                |                              | np.int8_, np.int16_, np.int32_, np.int64_, np.uint8_, np.uint16_,                   | ``'int8'``, ``'int16'``, ``'int32'``,       |
    |                        |                              | np.uint32_, np.uint64_                                                              | ``'int64'``, ``'uint8'``, ``'uint16'``,     |
    |                        |                              |                                                                                     | ``'uint32'``, ``'uint64'``                  |
    +------------------------+------------------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Float                  |                              | np.float32_, np.float64_                                                            | ``'float32'``, ``'float64'``                |
    +------------------------+------------------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Strings                |                              | `str <https://docs.python.org/3/library/stdtypes.html#str>`_                        | ``'string'``, ``'object'``                  |
    +------------------------+------------------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Datetime               |                              | np.datetime64_                                                                      | ``'datetime64[s]'``, ``'datetime64[ms]'``,  |
    |                        |                              |                                                                                     | ``'datetime64[us]'``, ``'datetime64[ns]'``  |
    +------------------------+------------------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Timedelta              |                              | np.timedelta64_                                                                     | ``'timedelta64[s]'``, ``'timedelta64[ms]'``,|
    | (duration type)        |                              |                                                                                     | ``'timedelta64[us]'``, ``'timedelta64[ns]'``|
    +------------------------+------------------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Categorical            | CategoricalDtype             | (none)                                                                              | ``'category'``                              |
    +------------------------+------------------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Boolean                |                              | np.bool_                                                                            | ``'bool'``                                  |
    +------------------------+------------------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Decimal                | Decimal64Dtype               | (none)                                                                              | (none)                                      |
    +------------------------+------------------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | List                   | ListDtype₁.₁                 | `list <https://docs.rapids.ai/api/cudf/stable/api.html#lists>`_                     | (none)₁.₂                                   |
    +------------------------+------------------------------+-------------------------------------------------------------------------------------+---------------------------------------------+
    | Struct                 | StructDtype₁.₁               | `struct <https://nvidia.github.io/spark-rapids/docs/supported_ops.html#types>`_     | (none)₁.₂                                   |
    +------------------------+------------------------------+-------------------------------------------------------------------------------------+---------------------------------------------+

**Note: All dtypes above are Nullable**

**1.1 The complete datatypes for both lists and structs are inferred from the data that it is composed upon. For example, the data type for the value** ``[[1, 2], [3, 4]]`` **is inferred as** ``ListDtype(int64)``.

**1.2 cuDF does not support string aliases where dtype would equal** ``'list'`` **or** ``'struct'`` **list's and structs are inferred by the datatype explicitly or through a determination based on the data that the list or struct is composed upon**


**Struct and List datatypes**

cuDF supports arbitrarily deep nested lists. Such as ``list(list(int))``, even list of structs or structs of lists

cuDF also supports arbitrary fields for structs - that is, it is possible to have a struct with any number of fields and any number of types that cuDF supports, even a struct of structs

Structs should be made up by same type of datatype or cuDF will produce an error - Example below
    
.. code-block:: python
    
    >>> df = cudf.Series(
    >>> [{'a':'dog', 'b':'cat', 'c':'astronomy'},
    >>> {'a':'fish', 'b':'gerbil', 'c':7}]
    >>> )
    >>> df
        
If Struct rows do not have the same members in each row, null values will be filled in for the members missing in any of the rows - Example below

.. code-block:: python

    >>> df = cudf.Series(
    >>> [{'a':'dog', 'b':'cat', 'c':'astronomy'},
    >>> {'a':'fish', 'b':'gerbil'}]
    >>> )
    >>> df
                                           Example
    0  {'a': 'dog ', 'b': 'cat', 'c': 'astronomy'}
    1      {'a': 'fish', 'b': 'gerbil', 'c': None}
 
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

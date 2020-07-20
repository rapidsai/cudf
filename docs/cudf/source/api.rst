API Reference
=============

.. currentmodule:: cudf.core.dataframe

DataFrame
---------
.. autoclass:: DataFrame
    :members:
    :inherited-members:
    :exclude-members: serialize, deserialize, device_deserialize, device_serialize, host_deserialize, host_serialize, to_dict

Series
------
.. currentmodule:: cudf.core.series

.. autoclass:: Series
    :members:
    :inherited-members:
    :exclude-members: serialize, deserialize, logical_not, logical_or, logical_and, remainder, sum_of_squares, fill, merge, iteritems, items, device_deserialize, device_serialize, host_deserialize, host_serialize, to_dict

Strings
-------
.. currentmodule:: cudf.core.column.string

.. autoclass:: StringMethods
    :members:

General Functions
-----------------
.. automodule:: cudf.core.reshape
    :members:
.. autofunction:: cudf.to_datetime


Index
-----
.. currentmodule:: cudf.core.index
.. autoclass:: Index
    :members:
    :inherited-members:
    :exclude-members: serialize, deserialize, device_deserialize, device_serialize, host_deserialize, host_serialize

RangeIndex
----------
.. currentmodule:: cudf.core.index
.. autoclass:: RangeIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

GenericIndex
------------
.. currentmodule:: cudf.core.index
.. autoclass:: GenericIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

MultiIndex
----------
.. currentmodule:: cudf.core.multiindex
.. autoclass:: MultiIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

Int8Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: Int8Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

Int16Index
----------
.. currentmodule:: cudf.core.index
.. autoclass:: Int16Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

Int32Index
----------
.. currentmodule:: cudf.core.index
.. autoclass:: Int32Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

Int64Index
----------
.. currentmodule:: cudf.core.index
.. autoclass:: Int64Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

UInt8Index
----------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt8Index
    :inherited-members:
    :members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

UInt16Index
-----------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt16Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

UInt32Index
-----------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt32Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

UInt64Index
-----------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt64Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

Float32Index
------------
.. currentmodule:: cudf.core.index
.. autoclass:: Float32Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

Float64Index
------------
.. currentmodule:: cudf.core.index
.. autoclass:: Float64Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

CategoricalIndex
----------------
.. currentmodule:: cudf.core.index
.. autoclass:: CategoricalIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

StringIndex
-----------
.. currentmodule:: cudf.core.index
.. autoclass:: StringIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

DatetimeIndex
-------------
.. currentmodule:: cudf.core.index
.. autoclass:: DatetimeIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize

Categories
----------
.. currentmodule:: cudf.core.column.categorical

.. autoclass:: CategoricalAccessor
    :members:

GroupBy
-------
.. currentmodule:: cudf.core.groupby.groupby

.. autoclass:: GroupBy
    :members:
    :exclude-members: deserialize, serialize

General utility functions
-------------------------
.. currentmodule:: cudf.testing

.. automodule:: cudf.testing.testing
    :members:



IO
--
.. currentmodule:: cudf.io

.. automodule:: cudf.io.csv
    :members:
.. automodule:: cudf.io.parquet
    :members:
.. automodule:: cudf.io.orc
    :members:
.. automodule:: cudf.io.json
    :members:
.. automodule:: cudf.io.avro
    :members:
.. automodule:: cudf.io.dlpack
    :members:
.. automodule:: cudf.io.feather
    :members:
.. automodule:: cudf.io.hdf
    :members:

GpuArrowReader
--------------
.. currentmodule:: cudf.comm.gpuarrow
.. autoclass:: GpuArrowReader
    :members:

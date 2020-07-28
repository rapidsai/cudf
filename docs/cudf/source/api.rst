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
    :exclude-members: serialize, deserialize, logical_not, logical_or, logical_and, remainder, sum_of_squares, fill, merge, iteritems, items, device_deserialize, device_serialize, host_deserialize, host_serialize, to_dict, tolist, to_list

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
    :exclude-members: serialize, deserialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

RangeIndex
----------
.. currentmodule:: cudf.core.index
.. autoclass:: RangeIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

GenericIndex
------------
.. currentmodule:: cudf.core.index
.. autoclass:: GenericIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

MultiIndex
----------
.. currentmodule:: cudf.core.multiindex
.. autoclass:: MultiIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

Int8Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: Int8Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

Int16Index
----------
.. currentmodule:: cudf.core.index
.. autoclass:: Int16Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

Int32Index
----------
.. currentmodule:: cudf.core.index
.. autoclass:: Int32Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

Int64Index
----------
.. currentmodule:: cudf.core.index
.. autoclass:: Int64Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

UInt8Index
----------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt8Index
    :inherited-members:
    :members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

UInt16Index
-----------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt16Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

UInt32Index
-----------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt32Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

UInt64Index
-----------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt64Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

Float32Index
------------
.. currentmodule:: cudf.core.index
.. autoclass:: Float32Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

Float64Index
------------
.. currentmodule:: cudf.core.index
.. autoclass:: Float64Index
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

CategoricalIndex
----------------
.. currentmodule:: cudf.core.index
.. autoclass:: CategoricalIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

StringIndex
-----------
.. currentmodule:: cudf.core.index
.. autoclass:: StringIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

DatetimeIndex
-------------
.. currentmodule:: cudf.core.index
.. autoclass:: DatetimeIndex
    :members:
    :inherited-members:
    :exclude-members: deserialize, serialize, device_deserialize, device_serialize, host_deserialize, host_serialize, tolist, to_list

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

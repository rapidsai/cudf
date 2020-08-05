Basics
======


Supported Dtypes
----------------

cuDF uses dtypes for Series or individual columns of a DataFrame. cuDF uses NumPy dtypes, NumPy provides support for ``float``, ``int``, ``bool``,
``'timedelta64[s]'``, ``'timedelta64[ms]'``, ``'timedelta64[us]'``, ``'timedelta64[ns]'``, ``'datetime64[s]'``, ``'datetime64[ms]'``,
``'datetime64[us]'``, ``'datetime64[ns]'`` (note that NumPy does not support timezone-aware datetimes).


The following table lists all of cudf types. For methods requiring dtype arguments, strings can be specified as indicated. See the respective documentation sections for more on each type.

.. csv-table:: Supported dtypes in cudf
   :file: dtypes.csv
   :widths: 30, 30, 40, 50
   :header-rows: 1
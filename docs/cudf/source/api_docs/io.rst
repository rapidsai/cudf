.. _api.io:

============
Input/output
============
.. currentmodule:: cudf

CSV
~~~
.. autosummary::
   :toctree: api/

   read_csv
   DataFrame.to_csv


.. currentmodule:: cudf.io.json

JSON
~~~~
.. autosummary::
   :toctree: api/

   read_json
   to_json

.. currentmodule:: cudf

Parquet
~~~~~~~
.. autosummary::
   :toctree: api/

   read_parquet
   DataFrame.to_parquet
   cudf.io.parquet.read_parquet_metadata

ORC
~~~
.. autosummary::
   :toctree: api/

   read_orc
   DataFrame.to_orc

.. currentmodule:: cudf

HDFStore: PyTables (HDF5)
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_hdf
   DataFrame.to_hdf

.. warning::

   HDF reader and writers are not GPU accelerated. These currently use CPU via Pandas.
   This may be GPU accelerated in the future.

Feather
~~~~~~~
.. autosummary::
   :toctree: api/

   read_feather
   DataFrame.to_feather

.. warning::

   Feather reader and writers are not GPU accelerated. These currently use CPU via Pandas.
   This may be GPU accelerated in the future.


API Reference
=============

.. currentmodule:: cudf.dataframe

DataFrame
---------
.. autoclass:: DataFrame
    :members:

..
  For cudf.concat function
..
.. automodule:: cudf.multi
    :members:

..
  For cudf.melt
..
.. automodule:: cudf.reshape.general
    :members:

Series
------
.. currentmodule:: cudf.dataframe.series

.. autoclass:: Series
    :members:
 
Groupby
-------
.. currentmodule:: cudf.groupby.groupby

.. autoclass:: Groupby
    :members:
..
  Sphinx explicit members (:members: apply, apply_grouped, as_df..), and exclude-members wasn't working
  Thus, manually specify the following for legacy_groupby.Groupby's docstring inclusion.
  Other methods in the class are legacy duplicates of cudf.groupby.groupby.Groupby
.. currentmodule:: cudf.groupby.legacy_groupby

.. automethod:: Groupby.apply
.. automethod:: Groupby.apply_grouped
.. automethod:: Groupby.as_df
.. automethod:: Groupby.std
.. automethod:: Groupby.var
.. automethod:: Groupby.sum_of_squares

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
.. automodule:: cudf.io.feather
    :members:
.. automodule:: cudf.io.hdf
    :members:

GpuArrowReader
--------------
.. currentmodule:: cudf.comm.gpuarrow
.. autoclass:: GpuArrowReader
    :members:

List handling
~~~~~~~~~~~~~

``Series.list`` can be used to access the values of the series as
lists and apply list methods to it. These can be accessed like
``Series.list.<function/property>``.

.. currentmodule:: cudf
.. autosummary::
   :toctree: api/

   Series.list

.. currentmodule:: cudf.core.column.lists.ListMethods
.. autosummary::
   :toctree: api/

   astype
   concat
   contains
   index
   get
   leaves
   len
   sort_values
   take
   unique

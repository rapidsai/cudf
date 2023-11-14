Struct handling
~~~~~~~~~~~~~~~

``Series.struct`` can be used to access the values of the series as
Structs and apply struct methods to it. These can be accessed like
``Series.struct.<function/property>``.

.. currentmodule:: cudf
.. autosummary::
   :toctree: api/

   Series.struct

.. currentmodule:: cudf.core.column.struct.StructMethods
.. autosummary::
   :toctree: api/

   field
   explode

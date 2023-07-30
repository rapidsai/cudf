================
Extension Dtypes
================
.. currentmodule:: cudf

cuDF supports a number of extension dtypes that build on top of the types that pandas supports. These dtypes are not directly available in pandas, which instead relies on object dtype arrays that run at Python rather than native speeds. The following dtypes are supported:


cudf.CategoricalDtype
=====================
.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

   CategoricalDtype


Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

    CategoricalDtype.categories
    CategoricalDtype.construct_from_string
    CategoricalDtype.deserialize
    CategoricalDtype.device_deserialize
    CategoricalDtype.device_serialize
    CategoricalDtype.empty
    CategoricalDtype.from_pandas
    CategoricalDtype.host_deserialize
    CategoricalDtype.host_serialize
    CategoricalDtype.is_dtype
    CategoricalDtype.kind
    CategoricalDtype.na_value
    CategoricalDtype.name
    CategoricalDtype.names
    CategoricalDtype.ordered
    CategoricalDtype.serialize
    CategoricalDtype.str
    CategoricalDtype.to_pandas
    CategoricalDtype.type


cudf.Decimal32Dtype
===================
.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

   Decimal32Dtype

Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

   Decimal32Dtype.ITEMSIZE
   Decimal32Dtype.MAX_PRECISION
   Decimal32Dtype.construct_array_type
   Decimal32Dtype.construct_from_string
   Decimal32Dtype.deserialize
   Decimal32Dtype.device_deserialize
   Decimal32Dtype.device_serialize
   Decimal32Dtype.empty
   Decimal32Dtype.from_arrow
   Decimal32Dtype.host_deserialize
   Decimal32Dtype.host_serialize
   Decimal32Dtype.is_dtype
   Decimal32Dtype.itemsize
   Decimal32Dtype.kind
   Decimal32Dtype.na_value
   Decimal32Dtype.name
   Decimal32Dtype.names
   Decimal32Dtype.precision
   Decimal32Dtype.scale
   Decimal32Dtype.serialize
   Decimal32Dtype.str
   Decimal32Dtype.to_arrow
   Decimal32Dtype.type

cudf.Decimal64Dtype
===================
.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

   Decimal64Dtype

Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

   Decimal64Dtype.ITEMSIZE
   Decimal64Dtype.MAX_PRECISION
   Decimal64Dtype.construct_array_type
   Decimal64Dtype.construct_from_string
   Decimal64Dtype.deserialize
   Decimal64Dtype.device_deserialize
   Decimal64Dtype.device_serialize
   Decimal64Dtype.empty
   Decimal64Dtype.from_arrow
   Decimal64Dtype.host_deserialize
   Decimal64Dtype.host_serialize
   Decimal64Dtype.is_dtype
   Decimal64Dtype.itemsize
   Decimal64Dtype.kind
   Decimal64Dtype.na_value
   Decimal64Dtype.name
   Decimal64Dtype.names
   Decimal64Dtype.precision
   Decimal64Dtype.scale
   Decimal64Dtype.serialize
   Decimal64Dtype.str
   Decimal64Dtype.to_arrow
   Decimal64Dtype.type

cudf.Decimal128Dtype
====================
.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

   Decimal128Dtype

Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

   Decimal128Dtype.ITEMSIZE
   Decimal128Dtype.MAX_PRECISION
   Decimal128Dtype.construct_array_type
   Decimal128Dtype.construct_from_string
   Decimal128Dtype.deserialize
   Decimal128Dtype.device_deserialize
   Decimal128Dtype.device_serialize
   Decimal128Dtype.empty
   Decimal128Dtype.from_arrow
   Decimal128Dtype.host_deserialize
   Decimal128Dtype.host_serialize
   Decimal128Dtype.is_dtype
   Decimal128Dtype.itemsize
   Decimal128Dtype.kind
   Decimal128Dtype.na_value
   Decimal128Dtype.name
   Decimal128Dtype.names
   Decimal128Dtype.precision
   Decimal128Dtype.scale
   Decimal128Dtype.serialize
   Decimal128Dtype.str
   Decimal128Dtype.to_arrow
   Decimal128Dtype.type

cudf.ListDtype
==============
.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

   ListDtype

Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

   ListDtype.construct_array_type
   ListDtype.construct_from_string
   ListDtype.deserialize
   ListDtype.device_deserialize
   ListDtype.device_serialize
   ListDtype.element_type
   ListDtype.empty
   ListDtype.from_arrow
   ListDtype.host_deserialize
   ListDtype.host_serialize
   ListDtype.is_dtype
   ListDtype.kind
   ListDtype.leaf_type
   ListDtype.na_value
   ListDtype.names
   ListDtype.serialize
   ListDtype.to_arrow
   ListDtype.type

cudf.StructDtype
================
.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

   StructDtype

Properties and Methods
----------------------
.. autosummary::
   :toctree: api/

   StructDtype.construct_array_type
   StructDtype.construct_from_string
   StructDtype.deserialize
   StructDtype.device_deserialize
   StructDtype.device_serialize
   StructDtype.empty
   StructDtype.fields
   StructDtype.from_arrow
   StructDtype.host_deserialize
   StructDtype.host_serialize
   StructDtype.is_dtype
   StructDtype.kind
   StructDtype.na_value
   StructDtype.names
   StructDtype.serialize
   StructDtype.to_arrow
   StructDtype.type

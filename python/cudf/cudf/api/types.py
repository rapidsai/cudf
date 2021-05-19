# Copyright (c) 2021, NVIDIA CORPORATION.
"""Define common type operations."""

from pandas.api import types as pd_types

# These methods are aliased directly into this namespace, but can be modified
# later if we determine that there is a need.

union_categoricals = pd_types.union_categoricals
infer_dtype = pd_types.infer_dtype
pandas_dtype = pd_types.pandas_dtype
is_bool_dtype = pd_types.is_bool_dtype
is_categorical_dtype = pd_types.is_categorical_dtype
is_complex_dtype = pd_types.is_complex_dtype
is_datetime64_any_dtype = pd_types.is_datetime64_any_dtype
is_datetime64_dtype = pd_types.is_datetime64_dtype
is_datetime64_ns_dtype = pd_types.is_datetime64_ns_dtype
is_datetime64tz_dtype = pd_types.is_datetime64tz_dtype
is_extension_type = pd_types.is_extension_type
is_extension_array_dtype = pd_types.is_extension_array_dtype
is_float_dtype = pd_types.is_float_dtype
is_int64_dtype = pd_types.is_int64_dtype
is_integer_dtype = pd_types.is_integer_dtype
is_interval_dtype = pd_types.is_interval_dtype
is_numeric_dtype = pd_types.is_numeric_dtype
is_object_dtype = pd_types.is_object_dtype
is_period_dtype = pd_types.is_period_dtype
is_signed_integer_dtype = pd_types.is_signed_integer_dtype
is_string_dtype = pd_types.is_string_dtype
is_timedelta64_dtype = pd_types.is_timedelta64_dtype
is_timedelta64_ns_dtype = pd_types.is_timedelta64_ns_dtype
is_unsigned_integer_dtype = pd_types.is_unsigned_integer_dtype
is_sparse = pd_types.is_sparse
is_dict_like = pd_types.is_dict_like
is_file_like = pd_types.is_file_like
is_list_like = pd_types.is_list_like
is_named_tuple = pd_types.is_named_tuple
is_iterator = pd_types.is_iterator
is_bool = pd_types.is_bool
is_categorical = pd_types.is_categorical
is_complex = pd_types.is_complex
is_float = pd_types.is_float
is_hashable = pd_types.is_hashable
is_integer = pd_types.is_integer
is_interval = pd_types.is_interval
is_number = pd_types.is_number
is_re = pd_types.is_re
is_re_compilable = pd_types.is_re_compilable
is_scalar = pd_types.is_scalar
is_dtype_equal = pd_types.is_dtype_equal

# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from functools import lru_cache

from numba import types
from numba.cuda.cudaimpl import lower as cuda_lower

from cudf.core.dtypes import dtype
from cudf.core.udf import api, utils

from . import (
    groupby_lowering,
    groupby_typing,
    masked_lowering,
    masked_typing,
    strings_lowering,
    strings_typing,
)

_units = ["ns", "ms", "us", "s"]
_datetime_cases = {types.NPDatetime(u) for u in _units}
_timedelta_cases = {types.NPTimedelta(u) for u in _units}
_supported_masked_types = (
    types.integer_domain
    | types.real_domain
    | _datetime_cases
    | _timedelta_cases
    | {types.boolean}
)


from cudf.core.udf.strings_typing import string_view, udf_string

cuda_lower(api.Masked, string_view, types.boolean)(
    masked_lowering.masked_constructor
)
_supported_masked_types |= {string_view, udf_string}


masked_typing._register_masked_constructor_typing(_supported_masked_types)
masked_lowering._register_masked_constructor_lowering(_supported_masked_types)

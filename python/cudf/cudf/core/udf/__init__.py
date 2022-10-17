# Copyright (c) 2022, NVIDIA CORPORATION.
import cupy as cp
import numpy as np
from numba import cuda, types
from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry,
)

import rmm

from cudf.core.column import as_column
from cudf.core.dtypes import dtype
from cudf.core.udf import api, row_function, utils
from cudf.utils.dtypes import STRING_TYPES

from . import masked_lowering, masked_typing

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
_STRING_UDFS_ENABLED = False
cudf_str_dtype = dtype(str)
try:
    import strings_udf

    if strings_udf.ENABLED:
        from . import strings_typing  # isort: skip
        from . import strings_lowering  # isort: skip
        from strings_udf import ptxpath
        from strings_udf._lib.cudf_jit_udf import (
            from_udf_string_array,
            to_string_view_array,
        )
        from strings_udf._typing import (
            str_view_arg_handler,
            string_view,
            udf_string,
        )

        _supported_masked_types |= {strings_typing.string_view}
        utils.launch_arg_getters[cudf_str_dtype] = to_string_view_array
        utils.output_col_getters[cudf_str_dtype] = from_udf_string_array
        utils.masked_array_types[cudf_str_dtype] = string_view
        row_function.itemsizes[cudf_str_dtype] = string_view.size_bytes

        utils.JIT_SUPPORTED_TYPES |= STRING_TYPES
        utils.ptx_files.append(ptxpath)
        utils.arg_handlers.append(str_view_arg_handler)
        utils.udf_return_type_map[string_view] = udf_string
        _STRING_UDFS_ENABLED = True
    else:
        del strings_udf

except ImportError as e:
    # allow cuDF to work without strings_udf
    pass

masked_typing._register_masked_constructor_typing(_supported_masked_types)
masked_lowering._register_masked_constructor_lowering(_supported_masked_types)

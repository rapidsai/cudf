# Copyright (c) 2022, NVIDIA CORPORATION.

from functools import lru_cache

from numba import types
from numba.cuda.cudaimpl import lower as cuda_lower

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
    from strings_udf import ptxpath

    if ptxpath:
        utils.ptx_files.append(ptxpath)

        from strings_udf._lib.cudf_jit_udf import (
            column_from_udf_string_array,
            column_to_string_view_array,
        )
        from strings_udf._typing import (
            str_view_arg_handler,
            string_view,
            udf_string,
        )

        from . import strings_typing  # isort: skip
        from . import strings_lowering  # isort: skip

        cuda_lower(api.Masked, string_view, types.boolean)(
            masked_lowering.masked_constructor
        )
        utils.JIT_SUPPORTED_TYPES |= STRING_TYPES
        _supported_masked_types |= {string_view, udf_string}

        @lru_cache(maxsize=None)
        def set_initial_malloc_heap_size():
            strings_udf.set_malloc_heap_size()

        def column_to_string_view_array_init_heap(col):
            # lazily allocate heap only when a string needs to be returned
            set_initial_malloc_heap_size()
            return column_to_string_view_array(col)

        utils.launch_arg_getters[
            cudf_str_dtype
        ] = column_to_string_view_array_init_heap
        utils.output_col_getters[cudf_str_dtype] = column_from_udf_string_array
        utils.masked_array_types[cudf_str_dtype] = string_view
        row_function.itemsizes[cudf_str_dtype] = string_view.size_bytes

        utils.arg_handlers.append(str_view_arg_handler)

        masked_typing.MASKED_INIT_MAP[udf_string] = udf_string

        _STRING_UDFS_ENABLED = True

except ImportError as e:
    # allow cuDF to work without strings_udf
    pass

masked_typing._register_masked_constructor_typing(_supported_masked_types)
masked_lowering._register_masked_constructor_lowering(_supported_masked_types)

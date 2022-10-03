# Copyright (c) 2022, NVIDIA CORPORATION.
import numpy as np
from numba import cuda, types
from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry,
)

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
try:
    import strings_udf

    if strings_udf.ENABLED:
        from . import strings_typing  # isort: skip
        from . import strings_lowering  # isort: skip
        from strings_udf import ptxpath
        from strings_udf._lib.cudf_jit_udf import to_string_view_array
        from strings_udf._typing import str_view_arg_handler, string_view

        # add an overload of MaskedType.__init__(string_view, bool)
        cuda_lower(api.Masked, strings_typing.string_view, types.boolean)(
            masked_lowering.masked_constructor
        )

        # add an overload of pack_return(string_view)
        cuda_lower(api.pack_return, strings_typing.string_view)(
            masked_lowering.pack_return_scalar_impl
        )

        _supported_masked_types |= {strings_typing.string_view}
        utils.launch_arg_getters[dtype("O")] = to_string_view_array
        utils.masked_array_types[dtype("O")] = string_view
        utils.JIT_SUPPORTED_TYPES |= STRING_TYPES
        utils.ptx_files.append(ptxpath)
        utils.arg_handlers.append(str_view_arg_handler)
        row_function.itemsizes[dtype("O")] = string_view.size_bytes

        _STRING_UDFS_ENABLED = True
    else:
        del strings_udf

except ImportError as e:
    # allow cuDF to work without strings_udf
    pass
masked_typing.register_masked_constructor(_supported_masked_types)

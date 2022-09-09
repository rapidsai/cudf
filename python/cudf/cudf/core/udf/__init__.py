# Copyright (c) 2022, NVIDIA CORPORATION.
from . import masked_typing, masked_lowering
from numba import cuda
from numba import types
from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry,
)
from cudf.core.udf import api
from cudf.core.udf import utils
from cudf.core.udf import row_function
from cudf.core.dtypes import dtype
import numpy as np

units = ["ns", "ms", "us", "s"]
datetime_cases = {types.NPDatetime(u) for u in units}
timedelta_cases = {types.NPTimedelta(u) for u in units}


supported_masked_types = (
    types.integer_domain
    | types.real_domain
    | datetime_cases
    | timedelta_cases
    | {types.boolean}
)

_STRING_UDFS_ENABLED = False
try:
    import strings_udf

    if strings_udf.ENABLED:
        from . import strings_typing
        from . import strings_lowering
        from strings_udf import ptxpath
        from strings_udf._typing import string_view, str_view_arg_handler
        from strings_udf._lib.cudf_jit_udf import to_string_view_array

        # add an overload of MaskedType.__init__(string_view, bool)
        cuda_lower(api.Masked, strings_typing.string_view, types.boolean)(
            masked_lowering.masked_constructor
        )

        # add an overload of pack_return(string_view)
        cuda_lower(api.pack_return, strings_typing.string_view)(
            masked_lowering.pack_return_scalar_impl
        )

        supported_masked_types |= {strings_typing.string_view}
        utils.launch_arg_getters[dtype("O")] = to_string_view_array
        utils.masked_array_types[dtype("O")] = string_view
        utils.files.append(ptxpath)
        utils.arg_handlers.append(str_view_arg_handler)
        row_function.itemsizes[dtype("O")] = string_view.size_bytes

        _STRING_UDFS_ENABLED = True
    else:
        del strings_udf

except ImportError:
    # allow cuDF to work without strings_udf
    pass

masked_typing.register_masked_constructor(supported_masked_types)

# Copyright (c) 2022, NVIDIA CORPORATION.
from . import masked_typing, masked_lowering
from numba import cuda
from numba import types
from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry,
)
from cudf.core.udf import api

driver_maj, driver_min = cuda.cudadrv.driver.get_version()
runtime_maj, runtime_min = cuda.cudadrv.runtime.runtime.get_version()

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
if driver_maj >= runtime_maj and driver_min >= runtime_min:
    from . import strings_typing
    from . import strings_lowering

    # add an overload of MaskedType.__init__(string_view, bool)
    cuda_lower(api.Masked, strings_typing.string_view, types.boolean)(
        masked_lowering.masked_constructor
    )

    # add an overload of pack_return(string_view)
    cuda_lower(api.pack_return, strings_typing.string_view)(
        masked_lowering.pack_return_scalar_impl
    )

    supported_masked_types |= {strings_typing.string_view}
    _STRING_UDFS_ENABLED = True


masked_typing.register_masked_constructor(supported_masked_types)

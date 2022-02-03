# Copyright (c) 2022, NVIDIA CORPORATION.
from numba import types
from numba.core.extending import typeof_impl
from numba.cuda.models import register_model, models
from cudf.core.buffer import Buffer
from numba.core.typing.templates import (
    AbstractTemplate,
)
from numba.core.typing import signature
from numba.core import cgutils
from cudf.core.udf.typing import MaskedType, StringView, string_view

from numba.cuda.cudadecl import registry as cuda_decl_registry
from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry)

from cudf_jit_udf import to_string_view_array
from numba import cuda

@cuda_decl_registry.register_global(len)
class MaskedStringViewLength(AbstractTemplate):
    """
    provide the length of a cudf::string_view like struct
    """
    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(args[0].value_type, StringView):
            return signature(types.int32, args[0])

_len_string_view = cuda.declare_device('len_2', types.int32(types.CPointer(string_view)))

def call_len_string_view(st):
    return _len_string_view(st)

@cuda_lower(len, MaskedType(types.pyobject))
def string_view_len_impl(context, builder, sig, args):
    retty = sig.return_type
    maskedty = sig.args[0]
    masked_str = cgutils.create_struct_proxy(maskedty)(
        context, builder, value=args[0]
    )

    # the first element is the string_view struct
    # get a pointer that we will copy the data to
    strty = masked_str.value.type
    arg = builder.alloca(strty)

    # store
    builder.store(masked_str.value, arg)
    
    result = context.compile_internal(
        builder,
        call_len_string_view,
        signature(retty, types.CPointer(string_view)),
        (arg,)
    )
    
    return result

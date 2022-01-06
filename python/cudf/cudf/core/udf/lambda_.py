import numba
import numpy as np
from numba import cuda, typeof
from numba.core.typing import signature
from numba.np import numpy_support
from numba.types import Tuple, boolean, int64, void

from cudf.core.udf.templates import lambda_kernel_template

import cudf
from cudf.core.udf.api import Masked
from cudf.core.udf.pipeline import (
    construct_signature,
    get_udf_return_type,
    mask_get,
    masked_array_type_from_col,
    pack_return,
)
from cudf.core.udf.typing import MaskedType
from cudf.utils import cudautils

def get_udf_return_type_series(sr, func, args=()):
    # args types are (typeof(input_col), typeof(arg0), typeof(arg1)...)
    compile_sig = (
        numpy_support.from_dtype(sr.dtype),
        *(typeof(arg) for arg in args)
    )
    _, output_type = cudautils.compile_udf(func, compile_sig)
    
    if not isinstance(output_type, MaskedType):
        numba_output_type = numpy_support.from_dtype(np.dtype(output_type))
    else:
        numba_output_type = output_type

    return (
        numba_output_type
        if not isinstance(numba_output_type, MaskedType)
        else numba_output_type.value_type
    )
    
    
def construct_signature_from_series(series, return_type, args):
    """
    Build the signature of numba types that will be used to
    actually JIT the kernel itself later, accounting for types
    and offsets. Skips columns with unsupported dtypes.
    """

    # Tuple of arrays, first the output data array, then the mask
    return_type = Tuple((return_type[::1], boolean[::1]))
    offsets = []
    sig = [return_type, int64]
    sig.append(masked_array_type_from_col(series._column))
    offsets.append(int64)

    # return_type, size, data, masks, offsets, extra args
    sig = void(*(sig + offsets + [typeof(arg) for arg in args]))

    return sig


def make_kernel(args=()):
    extra_args = ", ".join([f"extra_arg_{i}" for i in range(len(args))])
    return lambda_kernel_template.format(extra_args=extra_args)


def compile_or_get_lambda_udf(sr, func, args=()):
    udf_return_type = get_udf_return_type_series(sr, func, args)

    sig = construct_signature_from_series(sr, udf_return_type, args=args)

    f_ = cuda.jit(device=True)(func)

    local_exec_context = {}
    global_exec_context = {
        "f_": f_,
        "cuda": cuda,
        "Masked": Masked,
        "mask_get": mask_get,
        "pack_return": pack_return,
    }
    exec(
        make_kernel(args=args),
        global_exec_context,
        local_exec_context,
    )
    _kernel = local_exec_context["_kernel"]

    kernel = cuda.jit(sig)(_kernel)

    return kernel, numpy_support.as_dtype(udf_return_type)

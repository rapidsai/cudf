from numba import cuda
from numba.np import numpy_support

from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.templates import (
    lambda_kernel_template,
    masked_input_initializer_template,
    unmasked_input_initializer_template,
)
from cudf.core.udf.typing import MaskedType
from cudf.core.udf.utils import (
    construct_signature,
    get_udf_return_type,
    mask_get,
)


def make_kernel(sr, args):
    extra_args = ", ".join([f"extra_arg_{i}" for i in range(len(args))])

    masked_initializer = (
        masked_input_initializer_template
        if sr._column.mask
        else unmasked_input_initializer_template
    )
    masked_initializer = masked_initializer.format(idx=0)

    return lambda_kernel_template.format(
        extra_args=extra_args, masked_initializer=masked_initializer
    )


def compile_or_get_lambda_function(sr, func, *args):

    sr_type = MaskedType(numpy_support.from_dtype(sr.dtype))
    udf_return_type = get_udf_return_type(sr_type, func, args)
    sig = construct_signature(sr, udf_return_type, args=args)

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
        make_kernel(sr, args=args), global_exec_context, local_exec_context,
    )
    _kernel = local_exec_context["_kernel"]
    kernel = cuda.jit(sig)(_kernel)

    return kernel, numpy_support.as_dtype(udf_return_type)

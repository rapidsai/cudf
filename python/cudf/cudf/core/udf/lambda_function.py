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
    generate_cache_key,
    get_udf_return_type,
    mask_get,
    precompiled,
)


def make_kernel(sr, args):
    """
    Function to write numba kernels for `Series.apply` as a string.
    Workaround until numba supports functions that use `*args`

    `Series.apply` expects functions of a single variable and possibly
    one or more constants, such as:

    def f(x, c, k):
        return (x + c) / k

    where the `x` are meant to be the values of the series. Since there
    can be only one column, the only thing that varies in the kinds of
    kernels that we want is the number of extra_args

    def _kernel(retval, 
                size, 
                input_col_0, 
                offset_0,    
                extra_arg_0, # the extra arg `c`
                extra_arg_1, # the extra arg `k`
    ):
        i = cuda.grid(1)
        ret_data_arr, ret_mask_arr = retval
        
        if i < size:
            d_0, m_0 = input_col_0
            masked_0 = Masked(d_0[i], mask_get(m_0, i + offset_0))

            ret = f_(masked_0, extra_arg_0, extra_arg_1)

            ret_masked = pack_return(ret)
            ret_data_arr[i] = ret_masked.value
            ret_mask_arr[i] = ret_masked.valid

    """
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

    # check to see if we already compiled this function
    cache_key = generate_cache_key(sr, func)
    if precompiled.get(cache_key) is not None:
        kernel, masked_or_scalar = precompiled[cache_key]
        return kernel, masked_or_scalar

    sr_type = MaskedType(numpy_support.from_dtype(sr.dtype))

    scalar_return_type = get_udf_return_type(sr_type, func, args)

    sig = construct_signature(sr, scalar_return_type, args=args)
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
    np_return_type = numpy_support.as_dtype(scalar_return_type)

    precompiled[cache_key] = (kernel, np_return_type)

    return kernel, np_return_type

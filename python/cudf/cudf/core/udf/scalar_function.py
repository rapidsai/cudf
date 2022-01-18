from numba import cuda
from numba.np import numpy_support

from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.templates import (
    scalar_kernel_template,
    masked_input_initializer_template,
    unmasked_input_initializer_template,
)
from cudf.core.udf.typing import MaskedType
from cudf.core.udf.utils import (
    construct_signature,
    get_udf_return_type,
    mask_get,
)


def scalar_kernel_from_template(sr, args):
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
            masked_0 = Masked(d_0[i], mask_get(m_0, i + offset_0)

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
    ).format(idx=0)

    return scalar_kernel_template.format(
        extra_args=extra_args, masked_initializer=masked_initializer
    )


def get_scalar_kernel(sr, func, args):
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
        scalar_kernel_from_template(sr, args=args),
        global_exec_context,
        local_exec_context,
    )

    kernel_string = local_exec_context["_kernel"]
    kernel = cuda.jit(sig)(kernel_string)

    return kernel, scalar_return_type

import cupy
import numpy as np
from numba import cuda
from numba.np import numpy_support
from numba.types import Tuple, boolean, int64, void
from nvtx import annotate

import cudf
import cachetools
from cudf._lib.transform import bools_to_mask
from cudf.core.udf.classes import Masked
from cudf.core.udf.typing import MaskedType
from cudf.utils import cudautils

libcudf_bitmask_type = numpy_support.from_dtype(np.dtype("int32"))
mask_bitsize = np.dtype("int32").itemsize * 8
_kernel_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)


@annotate("NUMBA JIT", color="green", domain="cudf_python")
def compile_masked_udf(func, dtypes):
    """
    Generate an inlineable PTX function that will be injected into
    a variadic kernel inside libcudf

    assume all input types are `MaskedType(input_col.dtype)` and then
    compile the requestied PTX function as a function over those types
    """
    to_compiler_sig = tuple(
        MaskedType(arg)
        for arg in (numpy_support.from_dtype(np_type) for np_type in dtypes)
    )
    # Get the inlineable PTX function
    ptx, output_type = cudautils.compile_udf(func, to_compiler_sig)

    if not isinstance(output_type, MaskedType):
        numba_output_type = numpy_support.from_dtype(np.dtype(output_type))
    else:
        numba_output_type = output_type

    return numba_output_type, ptx


def nulludf(func):
    """
    Mimic pandas API:

    def f(x, y):
        return x + y
    df.apply(lambda row: f(row['x'], row['y']))

    in this scheme, `row` is actually the whole dataframe
    `DataFrame` sends `self` in as `row` and subsequently
    we end up calling `f` on the resulting columns since
    the dataframe is dict-like
    """

    def wrapper(*args):
        from cudf import DataFrame

        # This probably creates copies but is fine for now
        to_udf_table = DataFrame(
            {idx: arg for idx, arg in zip(range(len(args)), args)}
        )
        # Frame._apply
        return to_udf_table._apply(func)

    return wrapper


def masked_arrty_from_np_type(dtype):
    """
    Return a type representing a tuple of arrays,
    the first element an array of the numba type
    corresponding to `dtype`, and the second an
    array of bools represe
    """
    nb_scalar_ty = numpy_support.from_dtype(dtype)
    return Tuple((nb_scalar_ty[::1], libcudf_bitmask_type[::1]))


def construct_signature(df, return_type):
    """
    Build the signature of numba types that will be used to
    actually JIT the kernel itself later, accounting for types
    and offsets
    """

    # Tuple of arrays, first the output data array, then the mask
    return_type = Tuple((return_type[::1], boolean[::1]))
    offsets = []
    sig = [return_type]
    for col in df._data.values():
        sig.append(masked_arrty_from_np_type(col.dtype))
        offsets.append(int64)

    # return_type + data,masks + offsets + size
    sig = void(*(sig + offsets + [int64]))

    return sig


@cuda.jit(device=True)
def mask_get(mask, pos):
    return (mask[pos // mask_bitsize] >> (pos % mask_bitsize)) & 1


kernel_template = """\
def _kernel(retval, {input_columns}, {input_offsets}, size):
    i = cuda.grid(1)
    ret_data_arr, ret_mask_arr = retval
    if i < size:
{masked_input_initializers}
        ret = {user_udf_call}
        ret_data_arr[i] = {ret_value}
        ret_mask_arr[i] = {ret_valid}
"""

unmasked_input_initializer_template = """\
        d_{idx}, m_{idx} = input_col_{idx}
        masked_{idx} = Masked(d_{idx}[i], True)
"""

masked_input_initializer_template = """\
        d_{idx}, m_{idx} = input_col_{idx}
        masked_{idx} = Masked(d_{idx}[i], mask_get(m_{idx}, i + offset_{idx}))
"""

@annotate("DEFINE", color="yellow", domain="cudf_python")
def _define_function(df, scalar_return=False):
    # Create argument list for kernel
    input_columns = ", ".join(
        [f"input_col_{i}" for i in range(len(df.columns))]
    )

    input_offsets = ", ".join([f"offset_{i}" for i in range(len(df.columns))])

    # Create argument list to pass to device function
    args = ", ".join([f"masked_{i}" for i in range(len(df.columns))])
    user_udf_call = f"f_({args})"

    # Generate the initializers for each device function argument
    initializers = []
    for i, col in enumerate(df._data.values()):
        idx = str(i)
        if col.mask is not None:
            template = masked_input_initializer_template
        else:
            template = unmasked_input_initializer_template

        initializer = template.format(**{"idx": idx})

        initializers.append(initializer)

    masked_input_initializers = "\n".join(initializers)

    # Generate the code to extract the return value and mask from the device
    # function's return value depending on whether it's already masked
    if scalar_return:
        ret_value = "ret"
        ret_valid = "True"
    else:
        ret_value = "ret.value"
        ret_valid = "ret.valid"

    # Incorporate all of the above into the kernel code template
    d = {
        "input_columns": input_columns,
        "input_offsets": input_offsets,
        "masked_input_initializers": masked_input_initializers,
        "user_udf_call": user_udf_call,
        "ret_value": ret_value,
        "ret_valid": ret_valid,
    }

    return kernel_template.format(**d)


@annotate("UDF PIPELINE", color="black", domain="cudf_python")
def udf_pipeline(df, f):
    numba_return_type, ptx = compile_masked_udf(f, df.dtypes)
    _is_scalar_return = not isinstance(numba_return_type, MaskedType)
    scalar_return_type = (
        numba_return_type
        if _is_scalar_return
        else numba_return_type.value_type
    )

    sig = construct_signature(df, scalar_return_type)

    f_ = cuda.jit(device=True)(f)
    # Set f_launch into the global namespace
    lcl = {}
    exec(
        _define_function(df, scalar_return=_is_scalar_return),
        {"f_": f_, "cuda": cuda, "Masked": Masked, "mask_get": mask_get},
        lcl,
    )
    _kernel = lcl["_kernel"]

    # check to see if we already compiled this function
    kernel_cache_key = cudautils._make_cache_key(_kernel, sig)
    kernel_cache_key = (*kernel_cache_key, ptx)
    if _kernel_cache.get(kernel_cache_key) is None:
        kernel = cuda.jit(sig)(_kernel)
        _kernel_cache[kernel_cache_key] = kernel
    else:
        kernel = _kernel_cache[kernel_cache_key]

    ans_col = cupy.empty(
        len(df), dtype=numpy_support.as_dtype(scalar_return_type)
    )
    ans_mask = cudf.core.column.column_empty(len(df), dtype="bool")
    launch_args = [(ans_col, ans_mask)]
    offsets = []
    for col in df.columns:
        data = df[col]._column.data
        mask = df[col]._column.mask
        if mask is None:
            mask = cudf.core.buffer.Buffer()
        launch_args.append((data, mask))
        offsets.append(df[col]._column.offset)

    launch_args += offsets
    launch_args.append(len(df))  # size
    kernel.forall(len(df))(*launch_args)
    result = cudf.Series(ans_col).set_mask(bools_to_mask(ans_mask))
    return result

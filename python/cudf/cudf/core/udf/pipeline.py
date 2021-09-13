import cachetools
import numpy as np
from numba import cuda
from numba.np import numpy_support
from numba.types import Tuple, boolean, int64, void
from nvtx import annotate

from cudf.core.udf.classes import Masked
from cudf.core.udf.typing import MaskedType, pack_return
from cudf.utils import cudautils

libcudf_bitmask_type = numpy_support.from_dtype(np.dtype("int32"))
MASK_BITSIZE = np.dtype("int32").itemsize * 8
precompiled: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)

cuda.jit(device=True)(pack_return)


@annotate("NUMBA JIT", color="green", domain="cudf_python")
def compile_masked_udf(func, dtypes):
    """
    Compile a UDF with a signature of `MaskedType`s. Assumes a
    signature of `MaskedType(dtype)` for each dtype in `dtypes`.
    The UDFs logic (read from `func`s bytecode) is combined with
    the typing logic in `typing.py` to determine the UDFs output
    dtype and compile a string containing a PTX version of the
    the function.
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

    return numba_output_type


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
    array of bools representing a mask.
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
    return (mask[pos // MASK_BITSIZE] >> (pos % MASK_BITSIZE)) & 1


kernel_template = """\
def _kernel(retval, {input_columns}, {input_offsets}, size):
    i = cuda.grid(1)
    ret_data_arr, ret_mask_arr = retval
    if i < size:
{masked_input_initializers}
        ret = {user_udf_call}
        ret_masked = pack_return(ret)
        ret_data_arr[i] = ret_masked.value
        ret_mask_arr[i] = ret_masked.valid
"""

unmasked_input_initializer_template = """\
        d_{idx}, m_{idx} = input_col_{idx}
        masked_{idx} = Masked(d_{idx}[i], True)
"""

masked_input_initializer_template = """\
        d_{idx}, m_{idx} = input_col_{idx}
        masked_{idx} = Masked(d_{idx}[i], mask_get(m_{idx}, i + offset_{idx}))
"""


def _define_function(df, scalar_return=False):
    # Create argument list for kernel
    input_columns = ", ".join(
        [f"input_col_{i}" for i in range(len(df._data.items()))]
    )

    input_offsets = ", ".join([f"offset_{i}" for i in range(len(df._data))])

    # Create argument list to pass to device function
    args = ", ".join([f"masked_{i}" for i in range(len(df._data))])
    user_udf_call = f"f_({args})"

    # Generate the initializers for each device function argument
    initializers = []
    for i, col in enumerate(df._data.values()):
        idx = str(i)
        if col.mask is not None:
            template = masked_input_initializer_template
        else:
            template = unmasked_input_initializer_template

        initializer = template.format(idx=idx)

        initializers.append(initializer)

    masked_input_initializers = "\n".join(initializers)

    # Incorporate all of the above into the kernel code template
    d = {
        "input_columns": input_columns,
        "input_offsets": input_offsets,
        "masked_input_initializers": masked_input_initializers,
        "user_udf_call": user_udf_call,
    }

    return kernel_template.format(**d)


@annotate("UDF COMPILATION", color="darkgreen", domain="cudf_python")
def compile_or_get(df, f):
    """
    Return a compiled kernel in terms of MaskedTypes that launches a
    kernel equivalent of `f` for the dtypes of `df`. The kernel uses
    a thread for each row and calls `f` using that rows data / mask
    to produce an output value and output valdity for each row.

    If the UDF has already been compiled for this requested dtypes,
    a cached version will be returned instead of running compilation.

    """

    numba_return_type = compile_masked_udf(f, df.dtypes)
    _is_scalar_return = not isinstance(numba_return_type, MaskedType)
    scalar_return_type = (
        numba_return_type
        if _is_scalar_return
        else numba_return_type.value_type
    )

    sig = construct_signature(df, scalar_return_type)

    # check to see if we already compiled this function
    cache_key = cudautils._make_partial_cache_key(f)
    cache_key = (*cache_key, sig)
    if precompiled.get(cache_key) is not None:
        kernel = precompiled[cache_key]
    else:
        f_ = cuda.jit(device=True)(f)

        lcl = {}
        exec(
            # Defines a kernel named "_kernel" in the lcl dict
            _define_function(df, scalar_return=_is_scalar_return),
            {
                "f_": f_,
                "cuda": cuda,
                "Masked": Masked,
                "mask_get": mask_get,
                "pack_return": pack_return,
            },
            lcl,
        )
        # The python function definition representing the kernel
        _kernel = lcl["_kernel"]
        kernel = cuda.jit(sig)(_kernel)
        precompiled[cache_key] = kernel

    return kernel, numpy_support.as_dtype(scalar_return_type)

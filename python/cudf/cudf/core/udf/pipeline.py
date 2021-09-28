import math

import cachetools
import numpy as np
from numba import cuda
from numba.np import numpy_support
from numba.types import Record, Tuple, boolean, int64, void
from nvtx import annotate

from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.typing import MaskedType
from cudf.utils import cudautils

libcudf_bitmask_type = numpy_support.from_dtype(np.dtype("int32"))
MASK_BITSIZE = np.dtype("int32").itemsize * 8
precompiled: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)


@annotate("NUMBA JIT", color="green", domain="cudf_python")
def get_udf_return_type(func, df):
    """
    Get the return type of a masked UDF for a given set of argument dtypes. It
    is assumed that a `MaskedType(dtype)` is passed to the function for each
    input dtype.
    """
    np_struct_type = np.dtype(
        [(name, col.dtype) for name, col in df._data.items()]
    )
    row_type = masked_from_struct_dtype(np_struct_type)

    # Get the return type. The PTX is also returned by compile_udf, but is not
    # needed here.
    ptx, output_type = cudautils.compile_udf(func, (row_type,))
    if not isinstance(output_type, MaskedType):
        numba_output_type = numpy_support.from_dtype(np.dtype(output_type))
    else:
        numba_output_type = output_type

    return numba_output_type


def masked_array_type_from_col(col):
    """
    Return a type representing a tuple of arrays,
    the first element an array of the numba type
    corresponding to `dtype`, and the second an
    array of bools representing a mask.
    """
    nb_scalar_ty = numpy_support.from_dtype(col.dtype)
    if col.mask is None:
        return nb_scalar_ty[::1]
    else:
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
        sig.append(masked_array_type_from_col(col))
        offsets.append(int64)

    # return_type + data,masks + offsets + size
    sig = void(*(sig + offsets + [int64]))

    return sig


@cuda.jit(device=True)
def mask_get(mask, pos):
    return (mask[pos // MASK_BITSIZE] >> (pos % MASK_BITSIZE)) & 1


# utility function from numba
def _is_aligned_struct(struct):
    return struct.isalignedstruct


def masked_from_struct_dtype(dtype):
    """Convert a NumPy structured dtype to Numba Record type
    """
    if dtype.hasobject:
        raise TypeError("Do not support dtype containing object")

    fields = []
    offset = 0
    for name, info in dtype.fields.items():
        # *info* may have 3 element
        [elemdtype, _] = info[:2]
        title = info[2] if len(info) == 3 else None
        ty = numpy_support.from_dtype(elemdtype)
        infos = {
            "type": MaskedType(ty),
            "offset": offset,
            "title": title,
        }
        fields.append((name, infos))

        # increment offset by itemsize plus one byte for validity
        offset += elemdtype.itemsize + 1

        offset = int(math.ceil(offset / 8.0) * 8.0)

    # Note: dtype.alignment is not consistent.
    #       It is different after passing into a recarray.
    #       recarray(N, dtype=mydtype).dtype.alignment != mydtype.alignment
    return Record(fields, offset, True)


kernel_template = """\
def _kernel(retval, {input_columns}, {input_offsets}, size):
    i = cuda.grid(1)
    ret_data_arr, ret_mask_arr = retval
    if i < size:
        # Create a structured array with the desired fields
        rows = cuda.local.array(1, dtype=numba_rectype)

        # one element of that array
        row = rows[0]

{masked_input_initializers}
{row_initializers}

        # pass the abstract row into the udf
        ret = f_(row)

        # pack up the return values and set them
        ret_masked = pack_return(ret)
        ret_data_arr[i] = ret_masked.value
        ret_mask_arr[i] = ret_masked.valid
"""

unmasked_input_initializer_template = """\
        d_{idx} = input_col_{idx}
        masked_{idx} = Masked(d_{idx}[i], True)
"""

masked_input_initializer_template = """\
        d_{idx}, m_{idx} = input_col_{idx}
        masked_{idx} = Masked(d_{idx}[i], mask_get(m_{idx}, i + offset_{idx}))
"""

row_initializer_template = """\
        row["{name}"] = masked_{idx}
"""


def _define_function(df, scalar_return=False):
    # Create argument list for kernel
    input_columns = ", ".join([f"input_col_{i}" for i in range(len(df._data))])
    input_offsets = ", ".join([f"offset_{i}" for i in range(len(df._data))])

    # Generate the initializers for each device function argument
    initializers = []
    row_initializers = []
    for i, (colname, col) in enumerate(df._data.items()):
        idx = str(i)
        if col.mask is not None:
            template = masked_input_initializer_template
        else:
            template = unmasked_input_initializer_template

        initializer = template.format(idx=idx)

        initializers.append(initializer)

        row_initializer = row_initializer_template.format(
            idx=idx, name=colname
        )
        row_initializers.append(row_initializer)

    masked_input_initializers = "\n".join(initializers)
    row_initializers = "\n".join(row_initializers)

    # Incorporate all of the above into the kernel code template
    np_struct_type = np.dtype(
        [(name, col.dtype) for name, col in df._data.items()]
    )
    numba_rectype = masked_from_struct_dtype(np_struct_type)
    d = {
        "input_columns": input_columns,
        "input_offsets": input_offsets,
        "masked_input_initializers": masked_input_initializers,
        "row_initializers": row_initializers,
        "numba_rectype": numba_rectype,
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

    # check to see if we already compiled this function
    cache_key = (
        *cudautils.make_cache_key(f, tuple(df.dtypes)),
        *(col.mask is None for col in df._data.values()),
    )
    if precompiled.get(cache_key) is not None:
        kernel, scalar_return_type = precompiled[cache_key]
        return kernel, scalar_return_type

    numba_return_type = get_udf_return_type(f, df)
    _is_scalar_return = not isinstance(numba_return_type, MaskedType)
    scalar_return_type = (
        numba_return_type
        if _is_scalar_return
        else numba_return_type.value_type
    )

    sig = construct_signature(df, scalar_return_type)
    f_ = cuda.jit(device=True)(f)

    np_struct_type = np.dtype(
        [(name, col.dtype) for name, col in df._data.items()]
    )
    numba_rectype = masked_from_struct_dtype(np_struct_type)

    # Dict of 'local' variables into which `_kernel` is defined
    local_exec_context = {}
    global_exec_context = {
        "f_": f_,
        "cuda": cuda,
        "Masked": Masked,
        "mask_get": mask_get,
        "pack_return": pack_return,
        "numba_rectype": numba_rectype,
    }
    exec(
        _define_function(df, scalar_return=_is_scalar_return),
        global_exec_context,
        local_exec_context,
    )
    # The python function definition representing the kernel
    _kernel = local_exec_context["_kernel"]
    kernel = cuda.jit(sig)(_kernel)
    scalar_return_type = numpy_support.as_dtype(scalar_return_type)
    precompiled[cache_key] = (kernel, scalar_return_type)

    return kernel, scalar_return_type

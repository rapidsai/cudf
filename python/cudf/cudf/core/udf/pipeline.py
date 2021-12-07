import math
from typing import Callable

import cachetools
import numpy as np
from numba import cuda, typeof
from numba.np import numpy_support
from numba.types import Poison, Record, Tuple, boolean, int64, void
from nvtx import annotate

from cudf.core.dtypes import CategoricalDtype
from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.typing import MaskedType
from cudf.utils import cudautils
from cudf.utils.dtypes import (
    BOOL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
)

libcudf_bitmask_type = numpy_support.from_dtype(np.dtype("int32"))
MASK_BITSIZE = np.dtype("int32").itemsize * 8
precompiled: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)

JIT_SUPPORTED_TYPES = (
    NUMERIC_TYPES | BOOL_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES
)


def _is_jit_supported_type(dtype):
    # category dtype isn't hashable
    if isinstance(dtype, CategoricalDtype):
        return False
    return str(dtype) in JIT_SUPPORTED_TYPES


def all_dtypes_from_frame(frame):
    return {
        colname: col.dtype
        if _is_jit_supported_type(col.dtype)
        else np.dtype("O")
        for colname, col in frame._data.items()
    }


def supported_dtypes_from_frame(frame):
    return {
        colname: col.dtype
        for colname, col in frame._data.items()
        if _is_jit_supported_type(col.dtype)
    }


def supported_cols_from_frame(frame):
    return {
        colname: col
        for colname, col in frame._data.items()
        if _is_jit_supported_type(col.dtype)
    }


def generate_cache_key(frame, func: Callable):
    """Create a cache key that uniquely identifies a compilation.

    A new compilation is needed any time any of the following things change:
    - The UDF itself as defined in python by the user
    - The types of the columns utilized by the UDF
    - The existence of the input columns masks
    """
    return (
        *cudautils.make_cache_key(func, all_dtypes_from_frame(frame).values()),
        *(col.mask is None for col in frame._data.values()),
        *frame._data.keys(),
    )


def get_frame_row_type(dtype):
    """
    Get the numba `Record` type corresponding to a frame.
    Models each column and its mask as a MaskedType and
    models the row as a dictionary like data structure
    containing these MaskedTypes.

    Large parts of this function are copied with comments
    from the Numba internals and slightly modified to
    account for validity bools to be present in the final
    struct.
    """

    # Create the numpy structured type corresponding to the numpy dtype.

    fields = []
    offset = 0

    sizes = [val[0].itemsize for val in dtype.fields.values()]
    for i, (name, info) in enumerate(dtype.fields.items()):
        # *info* consists of the element dtype, its offset from the beginning
        # of the record, and an optional "title" containing metadata.
        # We ignore the offset in info because its value assumes no masking;
        # instead, we compute the correct offset based on the masked type.
        elemdtype = info[0]
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

        # Align the next member of the struct to be a multiple of the
        # memory access size, per PTX ISA 7.4/5.4.5
        if i < len(sizes) - 1:
            next_itemsize = sizes[i + 1]
            offset = int(math.ceil(offset / next_itemsize) * next_itemsize)

    # Numba requires that structures are aligned for the CUDA target
    _is_aligned_struct = True
    return Record(fields, offset, _is_aligned_struct)


@annotate("NUMBA JIT", color="green", domain="cudf_python")
def get_udf_return_type(frame, func: Callable, args=()):

    """
    Get the return type of a masked UDF for a given set of argument dtypes. It
    is assumed that the function consumes a dictionary whose keys are strings
    and whose values are of MaskedType. Initially assume that the UDF may be
    written to utilize any field in the row - including those containing an
    unsupported dtype. If an unsupported dtype is actually used in the function
    the compilation should fail at `compile_udf`. If compilation succeeds, one
    can infer that the function does not use any of the columns of unsupported
    dtype - meaning we can drop them going forward and the UDF will still end
    up getting fed rows containing all the fields it actually needs to use to
    compute the answer for that row.
    """

    # present a row containing all fields to the UDF and try and compile
    row_type = get_frame_row_type(
        np.dtype(list(all_dtypes_from_frame(frame).items()))
    )
    compile_sig = (row_type, *(typeof(arg) for arg in args))

    # Get the return type. The PTX is also returned by compile_udf, but is not
    # needed here.
    ptx, output_type = cudautils.compile_udf(func, compile_sig)
    if not isinstance(output_type, MaskedType):
        numba_output_type = numpy_support.from_dtype(np.dtype(output_type))
    else:
        numba_output_type = output_type

    return (
        numba_output_type
        if not isinstance(numba_output_type, MaskedType)
        else numba_output_type.value_type
    )


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


def construct_signature(frame, return_type, args):
    """
    Build the signature of numba types that will be used to
    actually JIT the kernel itself later, accounting for types
    and offsets. Skips columns with unsupported dtypes.
    """

    # Tuple of arrays, first the output data array, then the mask
    return_type = Tuple((return_type[::1], boolean[::1]))
    offsets = []
    sig = [return_type, int64]
    for col in supported_cols_from_frame(frame).values():
        sig.append(masked_array_type_from_col(col))
        offsets.append(int64)

    # return_type, size, data, masks, offsets, extra args
    sig = void(*(sig + offsets + [typeof(arg) for arg in args]))

    return sig


@cuda.jit(device=True)
def mask_get(mask, pos):
    return (mask[pos // MASK_BITSIZE] >> (pos % MASK_BITSIZE)) & 1


kernel_template = """\
def _kernel(retval, size, {input_columns}, {input_offsets}, {extra_args}):
    i = cuda.grid(1)
    ret_data_arr, ret_mask_arr = retval
    if i < size:
        # Create a structured array with the desired fields
        rows = cuda.local.array(1, dtype=row_type)

        # one element of that array
        row = rows[0]

{masked_input_initializers}
{row_initializers}

        # pass the assembled row into the udf
        ret = f_(row, {extra_args})

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


def _define_function(frame, row_type, args):
    """
    The kernel we want to JIT compile looks something like the following,
    which is an example for two columns that both have nulls present

    def _kernel(retval, input_col_0, input_col_1, offset_0, offset_1, size):
        i = cuda.grid(1)
        ret_data_arr, ret_mask_arr = retval
        if i < size:
            rows = cuda.local.array(1, dtype=row_type)
            row = rows[0]

            d_0, m_0 = input_col_0
            masked_0 = Masked(d_0[i], mask_get(m_0, i + offset_0))
            d_1, m_1 = input_col_1
            masked_1 = Masked(d_1[i], mask_get(m_1, i + offset_1))

            row["a"] = masked_0
            row["b"] = masked_1

            ret = f_(row)

            ret_masked = pack_return(ret)
            ret_data_arr[i] = ret_masked.value
            ret_mask_arr[i] = ret_masked.valid

    However we do not always have two columns and columns do not always have
    an associated mask. Ideally, we would just write one kernel and make use
    of `*args` - and then one function would work for any number of columns,
    currently numba does not support `*args` and treats functions it JITs as
    if `*args` is a singular argument. Thus we are forced to write the right
    functions dynamically at runtime and define them using `exec`.
    """
    # Create argument list for kernel
    frame = supported_cols_from_frame(frame)

    input_columns = ", ".join([f"input_col_{i}" for i in range(len(frame))])
    input_offsets = ", ".join([f"offset_{i}" for i in range(len(frame))])
    extra_args = ", ".join([f"extra_arg_{i}" for i in range(len(args))])

    # Generate the initializers for each device function argument
    initializers = []
    row_initializers = []
    for i, (colname, col) in enumerate(frame.items()):
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

    # Incorporate all of the above into the kernel code template
    d = {
        "input_columns": input_columns,
        "input_offsets": input_offsets,
        "extra_args": extra_args,
        "masked_input_initializers": "\n".join(initializers),
        "row_initializers": "\n".join(row_initializers),
        "numba_rectype": row_type,  # from global
    }

    return kernel_template.format(**d)


@annotate("UDF COMPILATION", color="darkgreen", domain="cudf_python")
def compile_or_get(frame, func, args):
    """
    Return a compiled kernel in terms of MaskedTypes that launches a
    kernel equivalent of `f` for the dtypes of `df`. The kernel uses
    a thread for each row and calls `f` using that rows data / mask
    to produce an output value and output validity for each row.

    If the UDF has already been compiled for this requested dtypes,
    a cached version will be returned instead of running compilation.

    CUDA kernels are void and do not return values. Thus, we need to
    preallocate a column of the correct dtype and pass it in as one of
    the kernel arguments. This creates a chicken-and-egg problem where
    we need the column type to compile the kernel, but normally we would
    be getting that type FROM compiling the kernel (and letting numba
    determine it as a return value). As a workaround, we compile the UDF
    itself outside the final kernel to invoke a full typing pass, which
    unfortunately is difficult to do without running full compilation.
    we then obtain the return type from that separate compilation and
    use it to allocate an output column of the right dtype.
    """

    # check to see if we already compiled this function
    cache_key = generate_cache_key(frame, func)
    if precompiled.get(cache_key) is not None:
        kernel, masked_or_scalar = precompiled[cache_key]
        return kernel, masked_or_scalar

    # precompile the user udf to get the right return type.
    # could be a MaskedType or a scalar type.
    scalar_return_type = get_udf_return_type(frame, func, args)

    # get_udf_return_type will throw a TypingError if the user tries to use
    # a field in the row containing an unsupported dtype, except in the
    # edge case where all the function does is return that element:

    # def f(row):
    #    return row[<bad dtype key>]
    # In this case numba is happy to return MaskedType(<bad dtype key>)
    # because it relies on not finding overloaded operators for types to raise
    # the exception, so we have to explicitly check for that case.
    if isinstance(scalar_return_type, Poison):
        raise TypeError(str(scalar_return_type))

    # this is the signature for the final full kernel compilation
    sig = construct_signature(frame, scalar_return_type, args)

    # this row type is used within the kernel to pack up the column and
    # mask data into the dict like data structure the user udf expects
    np_field_types = np.dtype(list(supported_dtypes_from_frame(frame).items()))
    row_type = get_frame_row_type(np_field_types)

    f_ = cuda.jit(device=True)(func)
    # Dict of 'local' variables into which `_kernel` is defined
    local_exec_context = {}
    global_exec_context = {
        "f_": f_,
        "cuda": cuda,
        "Masked": Masked,
        "mask_get": mask_get,
        "pack_return": pack_return,
        "row_type": row_type,
    }
    exec(
        _define_function(frame, row_type, args),
        global_exec_context,
        local_exec_context,
    )
    # The python function definition representing the kernel
    _kernel = local_exec_context["_kernel"]
    kernel = cuda.jit(sig)(_kernel)
    np_return_type = numpy_support.as_dtype(scalar_return_type)
    precompiled[cache_key] = (kernel, np_return_type)

    return kernel, np_return_type

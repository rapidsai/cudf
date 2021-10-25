import math

import cachetools
import numpy as np
from numba import cuda, typeof
from numba.np import numpy_support
from numba.types import Record, Tuple, boolean, int64, void
from nvtx import annotate

from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.typing import MaskedType
from cudf.utils import cudautils

libcudf_bitmask_type = numpy_support.from_dtype(np.dtype("int32"))
MASK_BITSIZE = np.dtype("int32").itemsize * 8
precompiled: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)


def get_frame_row_type(fr):
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

    # Create the numpy structured type corresponding to the frame.
    dtype = np.dtype([(name, col.dtype) for name, col in fr._data.items()])

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
def get_udf_return_type(func, df, args=()):
    """
    Get the return type of a masked UDF for a given set of argument dtypes. It
    is assumed that a `MaskedType(dtype)` is passed to the function for each
    input dtype.
    """
    # The users function args should be a row of the frame and then extra args
    row_type = get_frame_row_type(df)
    compile_sig = (row_type, *(typeof(arg) for arg in args))

    # Get the return type. The PTX is also returned by compile_udf, but is not
    # needed here.
    ptx, output_type = cudautils.compile_udf(func, compile_sig)
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


def construct_signature(df, return_type, args):
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
    sig = void(*(sig + offsets + [int64] + [typeof(arg) for arg in args]))

    return sig


@cuda.jit(device=True)
def mask_get(mask, pos):
    return (mask[pos // MASK_BITSIZE] >> (pos % MASK_BITSIZE)) & 1


kernel_template = """\
def _kernel(retval, {input_columns}, {input_offsets}, {extra_args}, size):
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


def _define_function(fr, row_type, args, scalar_return=False):
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
    funtions dynamically at runtime and define them using `exec`.
    """
    # Create argument list for kernel
    input_columns = ", ".join([f"input_col_{i}" for i in range(len(fr._data))])
    input_offsets = ", ".join([f"offset_{i}" for i in range(len(fr._data))])
    extra_args = ",".join("extra_arg_" + str(i) for i in range(len(args)))

    # Generate the initializers for each device function argument
    initializers = []
    row_initializers = []
    for i, (colname, col) in enumerate(fr._data.items()):
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
    d = {
        "input_columns": input_columns,
        "input_offsets": input_offsets,
        "extra_args": extra_args,
        "masked_input_initializers": masked_input_initializers,
        "row_initializers": row_initializers,
        "numba_rectype": row_type,  # from global
    }

    return kernel_template.format(**d)


@annotate("UDF COMPILATION", color="darkgreen", domain="cudf_python")
def compile_or_get(df, f, args):
    """
    Return a compiled kernel in terms of MaskedTypes that launches a
    kernel equivalent of `f` for the dtypes of `df`. The kernel uses
    a thread for each row and calls `f` using that rows data / mask
    to produce an output value and output valdity for each row.

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
    frame_dtypes = tuple(col.dtype for col in df._data.values())
    cache_key = (
        *cudautils.make_cache_key(f, frame_dtypes),
        *(col.mask is None for col in df._data.values()),
        *df._data.keys(),
    )
    if precompiled.get(cache_key) is not None:
        kernel, scalar_return_type = precompiled[cache_key]
        return kernel, scalar_return_type

    # precompile the user udf to get the right return type.
    # could be a MaskedType or a scalar type.
    numba_return_type = get_udf_return_type(f, df, args)

    _is_scalar_return = not isinstance(numba_return_type, MaskedType)
    scalar_return_type = (
        numba_return_type
        if _is_scalar_return
        else numba_return_type.value_type
    )

    # this is the signature for the final full kernel compilation
    sig = construct_signature(df, scalar_return_type, args)
    # this row type is used within the kernel to pack up the column and
    # mask data into the dict like data structure the user udf expects
    row_type = get_frame_row_type(df)

    f_ = cuda.jit(device=True)(f)
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
        _define_function(df, row_type, args, scalar_return=_is_scalar_return),
        global_exec_context,
        local_exec_context,
    )
    # The python function definition representing the kernel
    _kernel = local_exec_context["_kernel"]
    kernel = cuda.jit(sig)(_kernel)
    scalar_return_type = numpy_support.as_dtype(scalar_return_type)
    precompiled[cache_key] = (kernel, scalar_return_type)

    return kernel, scalar_return_type

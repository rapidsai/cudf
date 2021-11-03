import math
from typing import Callable

import cachetools
import numpy as np
from numba import cuda
from numba.np import numpy_support
from numba.types import Poison, Record, Tuple, boolean, int64, void
from nvtx import annotate

from cudf.core.udf._ops import _is_jit_supported_type
from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.typing import MaskedType
from cudf.utils import cudautils

libcudf_bitmask_type = numpy_support.from_dtype(np.dtype("int32"))
MASK_BITSIZE = np.dtype("int32").itemsize * 8
precompiled: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)


class _FrameJitMetadata(object):
    def __init__(self, frame):
        """
        convenience class that contains the metadata from the
        frame the UDF is targeting that is needed for jitting
        the eventual kernel.
        """
        self.all_dtypes = {}
        self.supported_dtypes = {}
        self.frame = frame

        for colname, col in self.frame._data.items():
            if _is_jit_supported_type(col.dtype):
                self.supported_dtypes[colname] = col.dtype
                np_type = col.dtype
            else:
                np_type = np.dtype("O")
            self.all_dtypes[colname] = np_type

        self.supported_cols = {
            name: self.frame._data[name]
            for name in self.supported_dtypes.keys()
        }


def generate_cache_key(frame_meta: _FrameJitMetadata, func: Callable):
    """
    Create a cache key that uniquely identifies a compilation. A new
    compilation is needed any time any of the following things change:
    - The UDF itself as defined in python by the user
    - The types of the columns actually logically utilized by the UDF
    - The existence of the input columns masks
    """
    cache_key = (
        *cudautils.make_cache_key(func, frame_meta.all_dtypes.values()),
        *(col.mask is None for col in frame_meta.frame._data.values()),
        *frame_meta.frame._data.keys(),
    )
    return cache_key


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
def get_udf_return_type(frame_meta: _FrameJitMetadata, func: Callable):
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
        np.dtype(list(frame_meta.all_dtypes.items()))
    )

    # Get the return type. The PTX is also returned by compile_udf, but is not
    # needed here.
    ptx, output_type = cudautils.compile_udf(func, (row_type,))
    if not isinstance(output_type, MaskedType):
        numba_output_type = numpy_support.from_dtype(np.dtype(output_type))
    else:
        numba_output_type = output_type

    scalar_return_type = (
        numba_output_type
        if not isinstance(numba_output_type, MaskedType)
        else numba_output_type.value_type
    )

    return scalar_return_type


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


def construct_signature(frame_meta: _FrameJitMetadata, return_type):
    """
    Build the signature of numba types that will be used to
    actually JIT the kernel itself later, accounting for types
    and offsets. Skips columns with unsupported dtypes.
    """

    # Tuple of arrays, first the output data array, then the mask
    return_type = Tuple((return_type[::1], boolean[::1]))
    offsets = []
    sig = [return_type]
    for col in frame_meta.supported_cols.values():
        sig.append(masked_array_type_from_col(col))
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
        # Create a structured array with the desired fields
        rows = cuda.local.array(1, dtype=row_type)

        # one element of that array
        row = rows[0]

{masked_input_initializers}
{row_initializers}

        # pass the assembled row into the udf
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


def _define_function(
    frame_meta: _FrameJitMetadata, row_type, scalar_return=False
):
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
    frame = {
        name: col
        for name, col in frame_meta.frame._data.items()
        if name in frame_meta.supported_dtypes.keys()
    }

    input_columns = ", ".join(f"input_col_{i}" for i in range(len(frame)))
    input_offsets = ", ".join(f"offset_{i}" for i in range(len(frame)))

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

    masked_input_initializers = "\n".join(initializers)
    row_initializer_list = "\n".join(row_initializers)

    # Incorporate all of the above into the kernel code template
    d = {
        "input_columns": input_columns,
        "input_offsets": input_offsets,
        "masked_input_initializers": masked_input_initializers,
        "row_initializers": row_initializer_list,
        "numba_rectype": row_type,  # from global
    }

    return kernel_template.format(**d)


def _check_return_type(ty):
    """
    In almost every case, get_udf_return_type will throw a typing error
    if the user tries to use a field in the row containing a dtype that
    is not supported. It does this by simply "not finding" overloads of
    any operators for the ancillary dtype we type that field as. But it
    is defeated by the following edge case:

    def f(row):
        return row[<bad dtype key>]

    In this case numba is happy to return MaskedType(<bad dtype key>),
    and won't throw because no operators were ever used. This function
    explicitly checks for that case.
    """
    if isinstance(ty, Poison):
        raise TypeError(str(ty))


@annotate("UDF COMPILATION", color="darkgreen", domain="cudf_python")
def compile_or_get(frame_meta, func):
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
    cache_key = generate_cache_key(frame_meta, func)
    if precompiled.get(cache_key) is not None:
        kernel, scalar_return_type = precompiled[cache_key]
        return kernel, scalar_return_type

    # precompile the user udf to get the right return type.
    # could be a MaskedType or a scalar type.
    scalar_return_type = get_udf_return_type(frame_meta, func)
    _check_return_type(scalar_return_type)
    _is_scalar_return = not isinstance(scalar_return_type, MaskedType)

    # this is the signature for the final full kernel compilation
    sig = construct_signature(frame_meta, scalar_return_type)

    # this row type is used within the kernel to pack up the column and
    # mask data into the dict like data structure the user udf expects
    np_field_types = np.dtype(list(frame_meta.supported_dtypes.items()))
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
        _define_function(
            frame_meta, row_type, scalar_return=_is_scalar_return
        ),
        global_exec_context,
        local_exec_context,
    )
    # The python function definition representing the kernel
    _kernel = local_exec_context["_kernel"]
    kernel = cuda.jit(sig)(_kernel)
    scalar_return_type = numpy_support.as_dtype(scalar_return_type)
    precompiled[cache_key] = (kernel, scalar_return_type)

    return kernel, scalar_return_type

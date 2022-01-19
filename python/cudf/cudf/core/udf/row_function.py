import math

import numpy as np
from numba import cuda
from numba.np import numpy_support
from numba.types import Record

from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.templates import (
    masked_input_initializer_template,
    row_initializer_template,
    row_kernel_template,
    unmasked_input_initializer_template,
)
from cudf.core.udf.typing import MaskedType
from cudf.core.udf.utils import (
    _all_dtypes_from_frame,
    _construct_signature,
    _get_kernel,
    _get_udf_return_type,
    _mask_get,
    _supported_cols_from_frame,
    _supported_dtypes_from_frame,
)


def _get_frame_row_type(dtype):
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


def _row_kernel_from_template(frame, row_type, args):
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
            masked_0 = Masked(d_0[i], _mask_get(m_0, i + offset_0))
            d_1, m_1 = input_col_1
            masked_1 = Masked(d_1[i], _mask_get(m_1, i + offset_1))

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
    frame = _supported_cols_from_frame(frame)

    input_columns = ", ".join([f"input_col_{i}" for i in range(len(frame))])
    input_offsets = ", ".join([f"offset_{i}" for i in range(len(frame))])
    extra_args = ", ".join([f"extra_arg_{i}" for i in range(len(args))])

    # Generate the initializers for each device function argument
    initializers = []
    row_initializers = []
    for i, (colname, col) in enumerate(frame.items()):
        idx = str(i)
        template = (
            masked_input_initializer_template
            if col.mask is not None
            else unmasked_input_initializer_template
        )
        initializers.append(template.format(idx=idx))
        row_initializers.append(
            row_initializer_template.format(
                idx=idx, name=colname
            )
        )

    return row_kernel_template.format(
        input_columns=input_columns,
        input_offsets=input_offsets,
        extra_args=extra_args,
        masked_input_initializers="\n".join(initializers),
        row_initializers="\n".join(row_initializers),
        numba_rectype=row_type,
    )


def _get_row_kernel(frame, func, args):
    row_type = _get_frame_row_type(
        np.dtype(list(_all_dtypes_from_frame(frame).items()))
    )
    scalar_return_type = _get_udf_return_type(row_type, func, args)

    # this is the signature for the final full kernel compilation
    sig = _construct_signature(frame, scalar_return_type, args)

    # this row type is used within the kernel to pack up the column and
    # mask data into the dict like data structure the user udf expects
    np_field_types = np.dtype(
        list(_supported_dtypes_from_frame(frame).items())
    )
    row_type = _get_frame_row_type(np_field_types)

    # Dict of 'local' variables into which `_kernel` is defined
    global_exec_context = {
        "cuda": cuda,
        "Masked": Masked,
        "_mask_get": _mask_get,
        "pack_return": pack_return,
        "row_type": row_type,
    }
    kernel_string = _row_kernel_from_template(frame, row_type, args)
    kernel = _get_kernel(kernel_string, global_exec_context, sig, func)

    return kernel, scalar_return_type

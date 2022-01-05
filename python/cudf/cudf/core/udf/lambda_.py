from numba import typeof
from numba.np import numpy_support
from cudf.utils import cudautils
from cudf.core.udf.typing import MaskedType
import numpy as np

kernel_template = """\
def _kernel(retval, size, dataarr, offset, {extra_args}):
    i = cuda.grid(1)
    ret_data_arr, ret_mask_arr = retval
    if i < size:
        data = dataarr[i]
{masked_input_initializer}

        ret = f_(data, {extra_args})
        ret_masked = pack_return(ret)
        ret_data_arr[i] = ret_masked.value
        ret_mask_arr[i] = ret_masked.valid


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

def get_udf_return_type(frame, func, args):
    """
    Return the numba return type of a lambda like
    function of scalars
    """
    col_type = numpy_support.from_dtype(frame._column.dtype)
    compile_sig = (col_type, *(typeof(arg) for arg in args))

    _, output_type = cudautils.compile_udf(func, compile_sig)

    if not isinstance(output_type, MaskedType):
        numba_output_type = numpy_support.from_dtype(np.dtype(output_type))
    else:
        numba_output_type = output_type

    return (
        numba_output_type
        if not isinstance(numba_output_type, MaskedType)
        else numba_output_type.value_type
    )
    
def _define_function(frame, row_type, args):
    """
    The kernel we want to JIT compile looks something like the following,
    which is an example for a column with nulls:

    def _kernel(retval, dataarr, offset, size, extra_args):
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

# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move, pair
from pylibcudf.libcudf cimport transform as cpp_transform
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport bitmask_type, size_type

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer

from .column cimport Column
from .gpumemoryview cimport gpumemoryview
from .types cimport DataType
from .utils cimport int_to_bitmask_ptr

__all__ = [
    "bools_to_mask",
    "compute_column",
    "encode",
    "mask_to_bools",
    "nans_to_nulls",
    "one_hot_encode",
    "transform",
]

cpdef tuple[gpumemoryview, int] nans_to_nulls(Column input):
    """Create a null mask preserving existing nulls and converting nans to null.

    For details, see :cpp:func:`nans_to_nulls`.

    Parameters
    ----------
    input : Column
        Column to produce new mask from.

    Returns
    -------
    Two-tuple of a gpumemoryview wrapping the null mask and the new null count.
    """
    cdef pair[unique_ptr[device_buffer], size_type] c_result

    with nogil:
        c_result = cpp_transform.nans_to_nulls(input.view())

    return (
        gpumemoryview(DeviceBuffer.c_from_unique_ptr(move(c_result.first))),
        c_result.second
    )


cpdef Column compute_column(Table input, Expression expr):
    """Create a column by evaluating an expression on a table.

    For details see :cpp:func:`compute_column`.

    Parameters
    ----------
    input : Table
        Table used for expression evaluation
    expr : Expression
        Expression to evaluate

    Returns
    -------
    Column of the evaluated expression
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_transform.compute_column(
            input.view(), dereference(expr.c_obj.get())
        )

    return Column.from_libcudf(move(c_result))


cpdef tuple[gpumemoryview, int] bools_to_mask(Column input):
    """Create a bitmask from a column of boolean elements

    Parameters
    ----------
    input : Column
        Column to produce new mask from.

    Returns
    -------
    tuple[gpumemoryview, int]
        Two-tuple of a gpumemoryview wrapping the bitmask and the null count.
    """
    cdef pair[unique_ptr[device_buffer], size_type] c_result

    with nogil:
        c_result = cpp_transform.bools_to_mask(input.view())

    return (
        gpumemoryview(DeviceBuffer.c_from_unique_ptr(move(c_result.first))),
        c_result.second
    )


cpdef Column mask_to_bools(Py_ssize_t bitmask, int begin_bit, int end_bit):
    """Creates a boolean column from given bitmask.

    Parameters
    ----------
    bitmask : int
        Pointer to the bitmask which needs to be converted
    begin_bit : int
        Position of the bit from which the conversion should start
    end_bit : int
        Position of the bit before which the conversion should stop

    Returns
    -------
    Column
        Boolean column of the bitmask from [begin_bit, end_bit]
    """
    cdef unique_ptr[column] c_result
    cdef bitmask_type * bitmask_ptr = int_to_bitmask_ptr(bitmask)

    with nogil:
        c_result = cpp_transform.mask_to_bools(bitmask_ptr, begin_bit, end_bit)

    return Column.from_libcudf(move(c_result))


cpdef Column transform(Column input, str unary_udf, DataType output_type, bool is_ptx):
    """Create a new column by applying a unary function against every
       element of an input column.

    Parameters
    ----------
    input : Column
        Column to transform.
    unary_udf : str
        The PTX/CUDA string of the unary function to apply.
    output_type : DataType
        The output type that is compatible with the output type in the unary_udf.
    is_ptx : bool
        If `True`, the UDF is treated as PTX code.
        If `False`, the UDF is treated as CUDA code.

    Returns
    -------
    Column
        The transformed column having the UDF applied to each element.
    """
    cdef unique_ptr[column] c_result
    cdef string c_unary_udf = unary_udf.encode()
    cdef bool c_is_ptx = is_ptx

    with nogil:
        c_result = cpp_transform.transform(
            input.view(), c_unary_udf, output_type.c_obj, c_is_ptx
        )

    return Column.from_libcudf(move(c_result))

cpdef tuple[Table, Column] encode(Table input):
    """Encode the rows of the given table as integers.

    Parameters
    ----------
    input : Table
        Table containing values to be encoded

    Returns
    -------
    tuple[Table, Column]
        The distinct row of the input table in sorted order,
        and a column of integer indices representing the encoded rows.
    """
    cdef pair[unique_ptr[table], unique_ptr[column]] c_result

    with nogil:
        c_result = cpp_transform.encode(input.view())

    return (
        Table.from_libcudf(move(c_result.first)),
        Column.from_libcudf(move(c_result.second))
    )

cpdef Table one_hot_encode(Column input, Column categories):
    """Encodes `input` by generating a new column
    for each value in `categories` indicating the presence
    of that value in `input`.

    Parameters
    ----------
    input : Column
        Column containing values to be encoded.
    categories : Column
        Column containing categories

    Returns
    -------
    Column
        A table of the encoded values.
    """
    cdef pair[unique_ptr[column], table_view] c_result
    cdef Table owner_table

    with nogil:
        c_result = cpp_transform.one_hot_encode(input.view(), categories.view())

    owner_table = Table(
        [Column.from_libcudf(move(c_result.first))] * c_result.second.num_columns()
    )

    return Table.from_table_view(c_result.second, owner_table)

# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from cython.operator cimport dereference

from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move, pair

from pylibcudf.libcudf cimport transform as cpp_transform
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport bitmask_type, size_type

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .gpumemoryview cimport gpumemoryview
from .types cimport DataType, null_aware
from .utils cimport _get_stream

__all__ = [
    "bools_to_mask",
    "compute_column",
    "encode",
    "mask_to_bools",
    "nans_to_nulls",
    "one_hot_encode",
    "transform",
]

cpdef tuple[gpumemoryview, int] nans_to_nulls(
    Column input,
    Stream stream=None,
):
    """Create a null mask preserving existing nulls and converting nans to null.

    For details, see :cpp:func:`nans_to_nulls`.

    Parameters
    ----------
    input : Column
        Column to produce new mask from.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Two-tuple of a gpumemoryview wrapping the null mask and the new null count.
    """
    cdef pair[unique_ptr[device_buffer], size_type] c_result

    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_transform.nans_to_nulls(input.view(), stream.view())

    return (
        gpumemoryview(DeviceBuffer.c_from_unique_ptr(move(c_result.first), stream)),
        c_result.second
    )


cpdef Column compute_column(Table input, Expression expr, Stream stream=None):
    """Create a column by evaluating an expression on a table.

    For details see :cpp:func:`compute_column`.

    Parameters
    ----------
    input : Table
        Table used for expression evaluation
    expr : Expression
        Expression to evaluate
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column of the evaluated expression
    """
    cdef unique_ptr[column] c_result

    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_transform.compute_column(
            input.view(), dereference(expr.c_obj.get()), stream.view()
        )

    return Column.from_libcudf(move(c_result), stream)


cpdef tuple[gpumemoryview, int] bools_to_mask(
    Column input,
    Stream stream=None,
):
    """Create a bitmask from a column of boolean elements

    Parameters
    ----------
    input : Column
        Column to produce new mask from.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    tuple[gpumemoryview, int]
        Two-tuple of a gpumemoryview wrapping the bitmask and the null count.
    """
    cdef pair[unique_ptr[device_buffer], size_type] c_result

    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_transform.bools_to_mask(input.view(), stream.view())

    return (
        gpumemoryview(DeviceBuffer.c_from_unique_ptr(move(c_result.first), stream)),
        c_result.second
    )


cpdef Column mask_to_bools(
    Py_ssize_t bitmask,
    int begin_bit,
    int end_bit,
    Stream stream=None,
):
    """Creates a boolean column from given bitmask.

    Parameters
    ----------
    bitmask : int
        Pointer to the bitmask which needs to be converted
    begin_bit : int
        Position of the bit from which the conversion should start
    end_bit : int
        Position of the bit before which the conversion should stop
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Boolean column of the bitmask from [begin_bit, end_bit]
    """
    cdef unique_ptr[column] c_result
    cdef bitmask_type * bitmask_ptr = <bitmask_type*>bitmask

    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_transform.mask_to_bools(
            bitmask_ptr,
            begin_bit,
            end_bit,
            stream.view(),
        )

    return Column.from_libcudf(move(c_result), stream)


cpdef Column transform(list[Column] inputs,
                       str transform_udf,
                       DataType output_type,
                       bool is_ptx,
                       null_aware is_null_aware,
                       Stream stream=None):
    """Create a new column by applying a transform function against
       multiple input columns.

    Parameters
    ----------
    inputs : list[Column]
        Columns to transform.
    transform_udf : str
        The PTX/CUDA string of the transform function to apply.
    output_type : DataType
        The output type that is compatible with the output type in the unary_udf.
    is_ptx : bool
        If `True`, the UDF is treated as PTX code.
        If `False`, the UDF is treated as CUDA code.
    is_null_aware: NullAware
        If `NO`, the UDF gets non-nullable parameters
        If `YES`, the UDF gets nullable parameters
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        The transformed column having the UDF applied to each element.
    """
    cdef vector[column_view] c_inputs
    cdef unique_ptr[column] c_result
    cdef string c_transform_udf = transform_udf.encode()
    cdef bool c_is_ptx = is_ptx
    cdef null_aware c_is_null_aware = is_null_aware
    cdef optional[void *] user_data

    stream = _get_stream(stream)

    for input in inputs:
        c_inputs.push_back((<Column?>input).view())

    with nogil:
        c_result = cpp_transform.transform(
            c_inputs,
            c_transform_udf,
            output_type.c_obj,
            c_is_ptx,
            user_data,
            c_is_null_aware,
            stream.view(),
        )

    return Column.from_libcudf(move(c_result), stream)

cpdef tuple[Table, Column] encode(Table input, Stream stream=None):
    """Encode the rows of the given table as integers.

    Parameters
    ----------
    input : Table
        Table containing values to be encoded
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    tuple[Table, Column]
        The distinct row of the input table in sorted order,
        and a column of integer indices representing the encoded rows.
    """
    cdef pair[unique_ptr[table], unique_ptr[column]] c_result

    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_transform.encode(input.view(), stream.view())

    return (
        Table.from_libcudf(move(c_result.first), stream),
        Column.from_libcudf(move(c_result.second), stream)
    )

cpdef Table one_hot_encode(
    Column input,
    Column categories,
    Stream stream=None,
):
    """Encodes `input` by generating a new column
    for each value in `categories` indicating the presence
    of that value in `input`.

    Parameters
    ----------
    input : Column
        Column containing values to be encoded.
    categories : Column
        Column containing categories
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        A table of the encoded values.
    """
    cdef pair[unique_ptr[column], table_view] c_result
    cdef Table owner_table

    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_transform.one_hot_encode(
            input.view(),
            categories.view(),
            stream.view(),
        )

    owner_table = Table(
        [Column.from_libcudf(move(c_result.first), stream)]
        * c_result.second.num_columns()
    )

    return Table.from_table_view(c_result.second, owner_table)

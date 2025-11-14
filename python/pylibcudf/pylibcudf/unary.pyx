# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf cimport unary as cpp_unary
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.unary cimport unary_operator
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.libcudf.unary import \
    unary_operator as UnaryOperator  # no-cython-lint

from .column cimport Column
from .types cimport DataType
from .utils cimport _get_stream, _get_memory_resource

__all__ = [
    "UnaryOperator",
    "cast",
    "is_nan",
    "is_not_nan",
    "is_null",
    "is_supported_cast",
    "is_valid",
    "unary_operation",
]

cpdef Column unary_operation(
    Column input, unary_operator op, Stream stream=None, DeviceMemoryResource mr=None
):
    """Perform a unary operation on a column.

    For details, see :cpp:func:`unary_operation`.

    Parameters
    ----------
    input : Column
        The column to operate on.
    op : UnaryOperator
        The operation to perform.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        The result of the unary operation
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        result = cpp_unary.unary_operation(input.view(), op, stream.view(), mr.get_mr())

    return Column.from_libcudf(move(result), stream, mr)


cpdef Column is_null(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """Check whether elements of a column are null.

    For details, see :cpp:func:`is_null`.

    Parameters
    ----------
    input : Column
        The column to check.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        A boolean column with ``True`` representing null values.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        result = cpp_unary.is_null(input.view(), stream.view(), mr.get_mr())

    return Column.from_libcudf(move(result), stream, mr)


cpdef Column is_valid(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """Check whether elements of a column are valid.

    For details, see :cpp:func:`is_valid`.

    Parameters
    ----------
    input : Column
        The column to check.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        A boolean column with ``True`` representing valid values.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        result = cpp_unary.is_valid(input.view(), stream.view(), mr.get_mr())

    return Column.from_libcudf(move(result), stream, mr)


cpdef Column cast(
    Column input, DataType data_type, Stream stream=None, DeviceMemoryResource mr=None
):
    """Cast a column to a different data type.

    For details, see :cpp:func:`cast`.

    Parameters
    ----------
    input : Column
        The column to check.
    data_type : DataType
        The data type to cast to.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        A boolean column with ``True`` representing null values.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        result = cpp_unary.cast(
            input.view(), data_type.c_obj, stream.view(), mr.get_mr()
        )

    return Column.from_libcudf(move(result), stream, mr)


cpdef Column is_nan(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """Check whether elements of a column are nan.

    For details, see :cpp:func:`is_nan`.

    Parameters
    ----------
    input : Column
        The column to check.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        A boolean column with ``True`` representing nan values.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        result = cpp_unary.is_nan(input.view(), stream.view(), mr.get_mr())

    return Column.from_libcudf(move(result), stream, mr)


cpdef Column is_not_nan(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """Check whether elements of a column are not nan.

    For details, see :cpp:func:`is_not_nan`.

    Parameters
    ----------
    input : Column
        The column to check.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        A boolean column with ``True`` representing non-nan values.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        result = cpp_unary.is_not_nan(input.view(), stream.view(), mr.get_mr())

    return Column.from_libcudf(move(result), stream, mr)

cpdef bool is_supported_cast(DataType from_, DataType to):
    """Check if a cast between datatypes is supported.

    For details, see :cpp:func:`is_supported_cast`.

    Parameters
    ----------
    from_
        The source datatype
    to
        The target datatype

    Returns
    -------
    bool
        True if the cast is supported.
    """
    with nogil:
        return cpp_unary.is_supported_cast(from_.c_obj, to.c_obj)

UnaryOperator.__str__ = UnaryOperator.__repr__

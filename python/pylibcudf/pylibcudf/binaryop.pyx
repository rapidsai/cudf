# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator import dereference

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf cimport binaryop as cpp_binaryop
from pylibcudf.libcudf.binaryop cimport binary_operator
from pylibcudf.libcudf.column.column cimport column

from pylibcudf.libcudf.binaryop import \
    binary_operator as BinaryOperator  # no-cython-lint

from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType

__all__ = ["BinaryOperator", "binary_operation", "is_supported_operation"]

cpdef Column binary_operation(
    LeftBinaryOperand lhs,
    RightBinaryOperand rhs,
    binary_operator op,
    DataType output_type
):
    """Perform a binary operation between a column and another column or scalar.

    ``lhs`` and ``rhs`` may be a
    :py:class:`~pylibcudf.column.Column` or a
    :py:class:`~pylibcudf.scalar.Scalar`, but at least one must be a
    :py:class:`~pylibcudf.column.Column`.

    For details, see :cpp:func:`binary_operation`.

    Parameters
    ----------
    lhs : Union[Column, Scalar]
        The left hand side argument.
    rhs : Union[Column, Scalar]
        The right hand side argument.
    op : BinaryOperator
        The operation to perform.
    output_type : DataType
        The data type to use for the output.

    Returns
    -------
    pylibcudf.Column
        The result of the binary operation
    """
    cdef unique_ptr[column] result

    if LeftBinaryOperand is Column and RightBinaryOperand is Column:
        with nogil:
            result = cpp_binaryop.binary_operation(
                lhs.view(),
                rhs.view(),
                op,
                output_type.c_obj
            )
    elif LeftBinaryOperand is Column and RightBinaryOperand is Scalar:
        with nogil:
            result = cpp_binaryop.binary_operation(
                lhs.view(),
                dereference(rhs.c_obj),
                op,
                output_type.c_obj
            )
    elif LeftBinaryOperand is Scalar and RightBinaryOperand is Column:
        with nogil:
            result = cpp_binaryop.binary_operation(
                dereference(lhs.c_obj),
                rhs.view(),
                op,
                output_type.c_obj
            )
    else:
        raise ValueError(f"Invalid arguments {lhs} and {rhs}")

    return Column.from_libcudf(move(result))


cpdef bool is_supported_operation(
    DataType out,
    DataType lhs,
    DataType rhs,
    binary_operator op
):
    """Check if an operation is supported for the given data types.

    For details, see :cpp:func::`is_supported_operation`.

    Parameters
    ----------
    out : DataType
        The output data type.
    lhs : DataType
        The left hand side data type.
    rhs : DataType
        The right hand side data type.
    op : BinaryOperator
        The operation to check.

    Returns
    -------
    bool
        True if the operation is supported, False otherwise
    """

    return cpp_binaryop.is_supported_operation(
        out.c_obj,
        lhs.c_obj,
        rhs.c_obj,
        op
    )

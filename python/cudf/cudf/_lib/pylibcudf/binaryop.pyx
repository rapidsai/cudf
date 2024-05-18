# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator import dereference

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf cimport binaryop as cpp_binaryop
from cudf._lib.pylibcudf.libcudf.binaryop cimport binary_operator
from cudf._lib.pylibcudf.libcudf.column.column cimport column

from cudf._lib.pylibcudf.libcudf.binaryop import \
    binary_operator as BinaryOperator  # no-cython-lint

from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType


cpdef Column binary_operation(
    LeftBinaryOperand lhs,
    RightBinaryOperand rhs,
    binary_operator op,
    DataType output_type
):
    """Perform a binary operation between a column and another column or scalar.

    ``lhs`` and ``rhs`` may be a
    :py:class:`~cudf._lib.pylibcudf.column.Column` or a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`, but at least one must be a
    :py:class:`~cudf._lib.pylibcudf.column.Column`.

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
            result = move(
                cpp_binaryop.binary_operation(
                    lhs.view(),
                    rhs.view(),
                    op,
                    output_type.c_obj
                )
            )
    elif LeftBinaryOperand is Column and RightBinaryOperand is Scalar:
        with nogil:
            result = move(
                cpp_binaryop.binary_operation(
                    lhs.view(),
                    dereference(rhs.c_obj),
                    op,
                    output_type.c_obj
                )
            )
    elif LeftBinaryOperand is Scalar and RightBinaryOperand is Column:
        with nogil:
            result = move(
                cpp_binaryop.binary_operation(
                    dereference(lhs.c_obj),
                    rhs.view(),
                    op,
                    output_type.c_obj
                )
            )
    else:
        raise ValueError(f"Invalid arguments {lhs} and {rhs}")

    return Column.from_libcudf(move(result))

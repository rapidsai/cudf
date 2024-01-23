# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator import dereference

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp cimport binaryop as cpp_binaryop
from cudf._lib.cpp.binaryop cimport binary_operator
from cudf._lib.cpp.column.column cimport column

from cudf._lib.cpp.binaryop import \
    binary_operator as BinaryOperator  # no-cython-lint

from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType


cpdef Column binary_operation(
    object lhs,
    object rhs,
    binary_operator op,
    DataType output_type
):
    """Perform a binary operation between a column and another column or scalar.

    Either ``lhs`` or ``rhs`` must be a
    :py:class:`~cudf._lib.pylibcudf.column.Column`. The other may be a
    :py:class:`~cudf._lib.pylibcudf.column.Column` or a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`.

    For details, see :cpp:func:`binary_operation`.

    Parameters
    ----------
    lhs : Column or Scalar
        The left hand side argument.
    rhs : Column or Scalar
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

    if isinstance(lhs, Column) and isinstance(rhs, Column):
        with nogil:
            result = move(
                cpp_binaryop.binary_operation(
                    (<Column> lhs).view(),
                    (<Column> rhs).view(),
                    op,
                    output_type.c_obj
                )
            )
    elif isinstance(lhs, Column) and isinstance(rhs, Scalar):
        with nogil:
            result = move(
                cpp_binaryop.binary_operation(
                    (<Column> lhs).view(),
                    dereference((<Scalar> rhs).c_obj),
                    op,
                    output_type.c_obj
                )
            )
    elif isinstance(lhs, Scalar) and isinstance(rhs, Column):
        with nogil:
            result = move(
                cpp_binaryop.binary_operation(
                    dereference((<Scalar> lhs).c_obj),
                    (<Column> rhs).view(),
                    op,
                    output_type.c_obj
                )
            )
    else:
        raise ValueError(f"Invalid arguments {lhs} and {rhs}")

    return Column.from_libcudf(move(result))

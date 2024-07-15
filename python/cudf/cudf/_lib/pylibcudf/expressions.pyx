# Copyright (c) 2024, NVIDIA CORPORATION.

import numpy as np

from cudf._lib.pylibcudf.libcudf.expressions import \
    ast_operator as ASTOperator  # no-cython-lint
from cudf._lib.pylibcudf.libcudf.expressions import \
    table_reference as TableReference  # no-cython-lint

from cython.operator cimport dereference
from libc.stdint cimport int64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf cimport expressions as libcudf_exp
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport (
    duration_scalar,
    numeric_scalar,
    string_scalar,
    timestamp_scalar,
)
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.pylibcudf.libcudf.wrappers.durations cimport (
    duration_ms,
    duration_ns,
    duration_s,
    duration_us,
)
from cudf._lib.pylibcudf.libcudf.wrappers.timestamps cimport (
    timestamp_ms,
    timestamp_ns,
    timestamp_s,
    timestamp_us,
)

# Aliases for simplicity
ctypedef unique_ptr[libcudf_exp.expression] expression_ptr

cdef class Literal(Expression):
    """
    A literal value used in an abstract syntax tree.

    For details, see :cpp:class:`cudf::ast::literal`.

    Parameters
    ----------
    value : Union[int, float, str, np.datetime64, np.timedelta64]
        A scalar value to use.
    """
    def __cinit__(self, value):
        if isinstance(value, int):
            self.c_scalar.reset(new numeric_scalar[int64_t](value, True))
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[int64_t] &>dereference(self.c_scalar)
            ))
        elif isinstance(value, float):
            self.c_scalar.reset(new numeric_scalar[double](value, True))
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[double] &>dereference(self.c_scalar)
            ))
        elif isinstance(value, str):
            self.c_scalar.reset(new string_scalar(value.encode(), True))
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <string_scalar &>dereference(self.c_scalar)
            ))
        elif isinstance(value, np.datetime64):
            scale, _ = np.datetime_data(value.dtype)
            int_value = value.astype(np.int64)
            if scale == "s":
                self.c_scalar.reset(new timestamp_scalar[timestamp_s](
                    <int64_t>int_value, True)
                )
                self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                    <timestamp_scalar[timestamp_s] &>dereference(self.c_scalar)
                ))
            elif scale == "ms":
                self.c_scalar.reset(new timestamp_scalar[timestamp_ms](
                    <int64_t>int_value, True)
                )
                self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                    <timestamp_scalar[timestamp_ms] &>dereference(self.c_scalar)
                ))
            elif scale == "us":
                self.c_scalar.reset(new timestamp_scalar[timestamp_us](
                    <int64_t>int_value, True)
                )
                self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                    <timestamp_scalar[timestamp_us] &>dereference(self.c_scalar)
                ))
            elif scale == "ns":
                self.c_scalar.reset(new timestamp_scalar[timestamp_ns](
                    <int64_t>int_value, True)
                )
                self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                    <timestamp_scalar[timestamp_ns] &>dereference(self.c_scalar)
                ))
            else:
                raise NotImplementedError(
                    f"Unhandled datetime scale {scale=}"
                )
        elif isinstance(value, np.timedelta64):
            scale, _ = np.datetime_data(value.dtype)
            int_value = value.astype(np.int64)
            if scale == "s":
                self.c_scalar.reset(new duration_scalar[duration_ms](
                    <int64_t>int_value, True)
                )
                self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                    <duration_scalar[duration_s] &>dereference(self.c_scalar)
                ))
            elif scale == "ms":
                self.c_scalar.reset(new duration_scalar[duration_ms](
                    <int64_t>int_value, True)
                )
                self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                    <duration_scalar[duration_ms] &>dereference(self.c_scalar)
                ))
            elif scale == "us":
                self.c_scalar.reset(new duration_scalar[duration_us](
                    <int64_t>int_value, True)
                )
                self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                    <duration_scalar[duration_us] &>dereference(self.c_scalar)
                ))
            elif scale == "ns":
                self.c_scalar.reset(new duration_scalar[duration_ns](
                    <int64_t>int_value, True)
                )
                self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                    <duration_scalar[duration_ns] &>dereference(self.c_scalar)
                ))
            else:
                raise NotImplementedError(
                    f"Unhandled timedelta scale {scale=}"
                )
        else:
            raise NotImplementedError(
                f"Don't know how to make literal with type {type(value)}"
            )


cdef class ColumnReference(Expression):
    """
    A expression referring to data from a column in a table.

    For details, see :cpp:class:`cudf::ast::column_reference`.

    Parameters
    ----------
    index : size_type
        The index of this column in the table
        (provided when the expression is evaluated).
    table_source : TableReference, default TableReferenece.LEFT
        Which table to use in cases with two tables (e.g. joins)
    """
    def __cinit__(
        self,
        size_type index,
        table_reference table_source=table_reference.LEFT
    ):
        self.c_obj = <expression_ptr>move(make_unique[libcudf_exp.column_reference](
            index, table_source
        ))


cdef class Operation(Expression):
    """
    An operation expression holds an operator and zero or more operands.

    For details, see :cpp:class:`cudf::ast::operation`.

    Parameters
    ----------
    op : Operator
    left : Expression
        Left input expression (left operand)
    right: Expression, default None
        Right input expression (right operand).
        You should only pass this if the input expression is a binary operation.
    """
    def __cinit__(self, ast_operator op, Expression left, Expression right=None):
        self.left = left
        self.right = right
        if right is None:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.operation](
                op, dereference(left.c_obj)
            ))
        else:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.operation](
                op, dereference(left.c_obj), dereference(right.c_obj)
            ))

cdef class ColumnNameReference(Expression):
    """
    A expression referring to data from a column in a table.

    For details, see :cpp:class:`cudf::ast::column_name_reference`.

    Parameters
    ----------
    column_name : str
        Name of this column in the table metadata
        (provided when the expression is evaluated).
    """
    def __cinit__(self, str name):
        self.c_obj = <expression_ptr> \
            move(make_unique[libcudf_exp.column_name_reference](
                <string>(name.encode("utf-8"))
            ))

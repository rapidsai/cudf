# Copyright (c) 2024, NVIDIA CORPORATION.
from pylibcudf.libcudf.expressions import \
    ast_operator as ASTOperator  # no-cython-lint
from pylibcudf.libcudf.expressions import \
    table_reference as TableReference  # no-cython-lint

from cython.operator cimport dereference
from libc.stdint cimport int32_t, int64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.libcudf cimport expressions as libcudf_exp
from pylibcudf.libcudf.scalar.scalar cimport (
    duration_scalar,
    numeric_scalar,
    string_scalar,
    timestamp_scalar,
)
from pylibcudf.libcudf.types cimport size_type, type_id
from pylibcudf.libcudf.wrappers.durations cimport (
    duration_ms,
    duration_ns,
    duration_s,
    duration_us,
)
from pylibcudf.libcudf.wrappers.timestamps cimport (
    timestamp_ms,
    timestamp_ns,
    timestamp_s,
    timestamp_us,
)

from .scalar cimport Scalar
from .traits cimport is_chrono, is_numeric
from .types cimport DataType

# Aliases for simplicity
ctypedef unique_ptr[libcudf_exp.expression] expression_ptr

cdef class Literal(Expression):
    """
    A literal value used in an abstract syntax tree.

    For details, see :cpp:class:`cudf::ast::literal`.

    Parameters
    ----------
    value : Scalar
        The Scalar value of the Literal.
        Must be either numeric, string, or a timestamp/duration scalar.
    """
    def __cinit__(self, Scalar value):
        self.scalar = value
        cdef DataType typ = value.type()
        cdef type_id tid = value.type().id()
        if not (is_numeric(typ) or is_chrono(typ) or tid == type_id.STRING):
            raise ValueError(
                "Only numeric, string, or timestamp/duration scalars are accepted"
            )
        # TODO: Accept type-erased scalar in AST C++ code
        # Then a lot of this code can be deleted
        if tid == type_id.INT64:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[int64_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.INT32:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[int32_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.FLOAT64:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[double] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.FLOAT32:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[float] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.STRING:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <string_scalar &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.TIMESTAMP_NANOSECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_ns] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.TIMESTAMP_MICROSECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_us] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.TIMESTAMP_MILLISECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_ms] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.TIMESTAMP_MILLISECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_ms] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.TIMESTAMP_SECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_s] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DURATION_NANOSECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_ns] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DURATION_MICROSECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_us] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DURATION_MILLISECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_ms] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DURATION_MILLISECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_ms] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.DURATION_SECONDS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_s] &>dereference(self.scalar.c_obj)
            ))
        else:
            raise NotImplementedError(
                f"Don't know how to make literal with type id {tid}"
            )

cdef class ColumnReference(Expression):
    """
    An expression referring to data from a column in a table.

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
    An expression referring to data from a column in a table.

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

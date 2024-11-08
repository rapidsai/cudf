# Copyright (c) 2024, NVIDIA CORPORATION.
import ast
import functools

import pyarrow as pa

from pylibcudf.libcudf.expressions import \
    ast_operator as ASTOperator  # no-cython-lint
from pylibcudf.libcudf.expressions import \
    table_reference as TableReference  # no-cython-lint

from cython.operator cimport dereference
from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libcpp cimport bool
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
    duration_D,
    duration_ms,
    duration_ns,
    duration_s,
    duration_us,
)
from pylibcudf.libcudf.wrappers.timestamps cimport (
    timestamp_D,
    timestamp_ms,
    timestamp_ns,
    timestamp_s,
    timestamp_us,
)

from .scalar cimport Scalar
from .traits cimport is_chrono, is_numeric
from .types cimport DataType

from .interop import from_arrow

# Aliases for simplicity
ctypedef unique_ptr[libcudf_exp.expression] expression_ptr

# Define this class just to have a docstring for it
cdef class Expression:
    """
    The base class for all expression types.
    This class cannot be instantiated directly, please
    instantiate one of its child classes instead.

    For details, see :cpp:class:`cudf::ast::expression`.
    """
    pass

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
        elif tid == type_id.INT16:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[int16_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.INT8:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[int8_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.UINT64:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[uint64_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.UINT32:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[uint32_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.UINT16:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[uint16_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.UINT8:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[uint8_t] &>dereference(self.scalar.c_obj)
            ))
        elif tid == type_id.BOOL8:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <numeric_scalar[bool] &>dereference(self.scalar.c_obj)
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
        elif tid == type_id.TIMESTAMP_DAYS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <timestamp_scalar[timestamp_D] &>dereference(self.scalar.c_obj)
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
        elif tid == type_id.DURATION_DAYS:
            self.c_obj = <expression_ptr> move(make_unique[libcudf_exp.literal](
                <duration_scalar[duration_D] &>dereference(self.scalar.c_obj)
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


# This dictionary encodes the mapping from Python AST operators to their cudf
# counterparts.
_python_cudf_operator_map = {
    # Binary operators
    ast.Add: ASTOperator.ADD,
    ast.Sub: ASTOperator.SUB,
    ast.Mult: ASTOperator.MUL,
    ast.Div: ASTOperator.DIV,
    ast.FloorDiv: ASTOperator.FLOOR_DIV,
    ast.Mod: ASTOperator.PYMOD,
    ast.Pow: ASTOperator.POW,
    ast.Eq: ASTOperator.EQUAL,
    ast.NotEq: ASTOperator.NOT_EQUAL,
    ast.Lt: ASTOperator.LESS,
    ast.Gt: ASTOperator.GREATER,
    ast.LtE: ASTOperator.LESS_EQUAL,
    ast.GtE: ASTOperator.GREATER_EQUAL,
    ast.BitXor: ASTOperator.BITWISE_XOR,
    # TODO: The mapping of logical/bitwise operators here is inconsistent with
    # pandas. In pandas, Both `BitAnd` and `And` map to
    # `ASTOperator.LOGICAL_AND` for booleans, while they map to
    # `ASTOperator.BITWISE_AND` for integers. However, there is no good way to
    # encode this at present because expressions can be arbitrarily nested so
    # we won't know the dtype of the input without inserting a much more
    # complex traversal of the expression tree to determine the output types at
    # each node. For now, we'll rely on users to use the appropriate operator.
    ast.BitAnd: ASTOperator.BITWISE_AND,
    ast.BitOr: ASTOperator.BITWISE_OR,
    ast.And: ASTOperator.LOGICAL_AND,
    ast.Or: ASTOperator.LOGICAL_OR,
    # Unary operators
    ast.Invert: ASTOperator.BIT_INVERT,
    ast.Not: ASTOperator.NOT,
    # TODO: Missing USub, possibility other unary ops?
}


# Mapping between Python function names encode in an ast.Call node and the
# corresponding libcudf C++ AST operators.
_python_cudf_function_map = {
    # TODO: Operators listed on
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#expression-evaluation-via-eval  # noqa: E501
    # that we don't support yet:
    # expm1, log1p, arctan2 and log10.
    "isnull": ASTOperator.IS_NULL,
    "isna": ASTOperator.IS_NULL,
    "sin": ASTOperator.SIN,
    "cos": ASTOperator.COS,
    "tan": ASTOperator.TAN,
    "arcsin": ASTOperator.ARCSIN,
    "arccos": ASTOperator.ARCCOS,
    "arctan": ASTOperator.ARCTAN,
    "sinh": ASTOperator.SINH,
    "cosh": ASTOperator.COSH,
    "tanh": ASTOperator.TANH,
    "arcsinh": ASTOperator.ARCSINH,
    "arccosh": ASTOperator.ARCCOSH,
    "arctanh": ASTOperator.ARCTANH,
    "exp": ASTOperator.EXP,
    "log": ASTOperator.LOG,
    "sqrt": ASTOperator.SQRT,
    "abs": ASTOperator.ABS,
    "ceil": ASTOperator.CEIL,
    "floor": ASTOperator.FLOOR,
    # TODO: Operators supported by libcudf with no Python function analog.
    # ast.rint: ASTOperator.RINT,
    # ast.cbrt: ASTOperator.CBRT,
}


class libcudfASTVisitor(ast.NodeVisitor):
    """A NodeVisitor specialized for constructing a libcudf expression tree.

    This visitor is designed to handle AST nodes that have libcudf equivalents.
    It constructs column references from names and literals from constants,
    then builds up operations. The final result can be accessed using the
    `expression` property. The visitor must be kept in scope for as long as the
    expression is needed because all of the underlying libcudf expressions will
    be destroyed when the libcudfASTVisitor is.

    Parameters
    ----------
    col_names : Tuple[str]
        The column names used to map the names in an expression.
    """

    def __init__(self, tuple col_names):
        self.stack = []
        self.nodes = []
        self.col_names = col_names

    @property
    def expression(self):
        """Expression: The result of parsing an AST."""
        assert len(self.stack) == 1
        return self.stack[-1]

    def visit_Name(self, node):
        try:
            col_id = self.col_names.index(node.id)
        except ValueError:
            raise ValueError(f"Unknown column name {node.id}")
        self.stack.append(ColumnReference(col_id))

    def visit_Constant(self, node):
        if not isinstance(node.value, (float, int, str, complex)):
            raise ValueError(
                f"Unsupported literal {repr(node.value)} of type "
                "{type(node.value).__name__}"
            )
        self.stack.append(
            Literal(from_arrow(pa.scalar(node.value)))
        )

    def visit_UnaryOp(self, node):
        self.visit(node.operand)
        self.nodes.append(self.stack.pop())
        if isinstance(node.op, ast.USub):
            # TODO: Except for leaf nodes, we won't know the type of the
            # operand, so there's no way to know whether this should be a float
            # or an int. We should maybe see what Spark does, and this will
            # probably require casting.
            self.nodes.append(Literal(from_arrow(pa.scalar(-1))))
            op = ASTOperator.MUL
            self.stack.append(Operation(op, self.nodes[-1], self.nodes[-2]))
        elif isinstance(node.op, ast.UAdd):
            self.stack.append(self.nodes[-1])
        else:
            op = _python_cudf_operator_map[type(node.op)]
            self.stack.append(Operation(op, self.nodes[-1]))

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        self.nodes.append(self.stack.pop())
        self.nodes.append(self.stack.pop())

        op = _python_cudf_operator_map[type(node.op)]
        self.stack.append(Operation(op, self.nodes[-1], self.nodes[-2]))

    def _visit_BoolOp_Compare(self, operators, operands, has_multiple_ops):
        # Helper function handling the common components of parsing BoolOp and
        # Compare AST nodes. These two types of nodes both support chaining
        # (e.g. `a > b > c` is equivalent to `a > b and b > c`, so this
        # function helps standardize that.

        # TODO: Whether And/Or and BitAnd/BitOr actually correspond to
        # logical or bitwise operators depends on the data types that they
        # are applied to. We'll need to add logic to map to that.
        inner_ops = []
        for op, (left, right) in zip(operators, operands):
            # Note that this will lead to duplicate nodes, e.g. if
            # the comparison is `a < b < c` that will be encoded as
            # `a < b and b < c`. We could potentially optimize by caching
            # expressions by name so that we only construct them once.
            self.visit(left)
            self.visit(right)

            self.nodes.append(self.stack.pop())
            self.nodes.append(self.stack.pop())

            op = _python_cudf_operator_map[type(op)]
            inner_ops.append(Operation(op, self.nodes[-1], self.nodes[-2]))

        self.nodes.extend(inner_ops)

        # If we have more than one comparator, we need to link them
        # together with LOGICAL_AND operators.
        if has_multiple_ops:
            op = ASTOperator.LOGICAL_AND

            def _combine_compare_ops(left, right):
                self.nodes.append(Operation(op, left, right))
                return self.nodes[-1]

            functools.reduce(_combine_compare_ops, inner_ops)

        self.stack.append(self.nodes[-1])

    def visit_BoolOp(self, node):
        operators = [node.op] * (len(node.values) - 1)
        operands = zip(node.values[:-1], node.values[1:])
        self._visit_BoolOp_Compare(operators, operands, len(node.values) > 2)

    def visit_Compare(self, node):
        operands = (node.left, *node.comparators)
        has_multiple_ops = len(operands) > 2
        operands = zip(operands[:-1], operands[1:])
        self._visit_BoolOp_Compare(node.ops, operands, has_multiple_ops)

    def visit_Call(self, node):
        try:
            op = _python_cudf_function_map[node.func.id]
        except KeyError:
            raise ValueError(f"Unsupported function {node.func}.")
        # Assuming only unary functions are supported, which is checked above.
        if len(node.args) != 1 or node.keywords:
            raise ValueError(
                f"Function {node.func} only accepts one positional "
                "argument."
            )
        self.visit(node.args[0])

        self.nodes.append(self.stack.pop())
        self.stack.append(Operation(op, self.nodes[-1]))


@functools.lru_cache(256)
def compute_column_expression(str expr, tuple col_names):
    """
    Create an expression for `pylibcudf.transform.compute_column`.

    Parameters
    ----------
    expr : str
        The expression to evaluate.

    col_names : tuple[str]
        The names associated with each column.

    Returns
    -------
    Expression
        Cached Expression for the given expr and col_names
    """
    visitor = libcudfASTVisitor(col_names)
    visitor.visit(ast.parse(expr))
    return visitor.expression

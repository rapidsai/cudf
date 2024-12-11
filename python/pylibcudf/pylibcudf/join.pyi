# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.expressions import Expression
from pylibcudf.table import Table
from pylibcudf.types import NullEquality

def inner_join(
    left_keys: Table, right_keys: Table, nulls_equal: NullEquality
) -> tuple[Column, Column]: ...
def left_join(
    left_keys: Table, right_keys: Table, nulls_equal: NullEquality
) -> tuple[Column, Column]: ...
def full_join(
    left_keys: Table, right_keys: Table, nulls_equal: NullEquality
) -> tuple[Column, Column]: ...
def left_semi_join(
    left_keys: Table, right_keys: Table, nulls_equal: NullEquality
) -> Column: ...
def left_anti_join(
    left_keys: Table, right_keys: Table, nulls_equal: NullEquality
) -> Column: ...
def cross_join(left: Table, right: Table) -> Table: ...
def conditional_inner_join(
    left: Table, right: Table, binary_predicate: Expression
) -> tuple[Column, Column]: ...
def conditional_left_join(
    left: Table, right: Table, binary_predicate: Expression
) -> tuple[Column, Column]: ...
def conditional_full_join(
    left: Table, right: Table, binary_predicate: Expression
) -> tuple[Column, Column]: ...
def conditional_left_semi_join(
    left: Table, right: Table, binary_predicate: Expression
) -> Column: ...
def conditional_left_anti_join(
    left: Table, right: Table, binary_predicate: Expression
) -> Column: ...
def mixed_inner_join(
    left_keys: Table,
    right_keys: Table,
    left_conditional: Table,
    right_conditional: Table,
    binary_predicate: Expression,
    nulls_equal: NullEquality,
) -> tuple[Column, Column]: ...
def mixed_left_join(
    left_keys: Table,
    right_keys: Table,
    left_conditional: Table,
    right_conditional: Table,
    binary_predicate: Expression,
    nulls_equal: NullEquality,
) -> tuple[Column, Column]: ...
def mixed_full_join(
    left_keys: Table,
    right_keys: Table,
    left_conditional: Table,
    right_conditional: Table,
    binary_predicate: Expression,
    nulls_equal: NullEquality,
) -> tuple[Column, Column]: ...
def mixed_left_semi_join(
    left_keys: Table,
    right_keys: Table,
    left_conditional: Table,
    right_conditional: Table,
    binary_predicate: Expression,
    nulls_equal: NullEquality,
) -> Column: ...
def mixed_left_anti_join(
    left_keys: Table,
    right_keys: Table,
    left_conditional: Table,
    right_conditional: Table,
    binary_predicate: Expression,
    nulls_equal: NullEquality,
) -> Column: ...

# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm import Stream
from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.expressions import Expression
from pylibcudf.table import Table
from pylibcudf.types import NullEquality

def inner_join(
    left_keys: Table,
    right_keys: Table,
    nulls_equal: NullEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Column, Column]: ...
def left_join(
    left_keys: Table,
    right_keys: Table,
    nulls_equal: NullEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Column, Column]: ...
def full_join(
    left_keys: Table,
    right_keys: Table,
    nulls_equal: NullEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Column, Column]: ...
def left_semi_join(
    left_keys: Table,
    right_keys: Table,
    nulls_equal: NullEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def left_anti_join(
    left_keys: Table,
    right_keys: Table,
    nulls_equal: NullEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def cross_join(
    left: Table,
    right: Table,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def conditional_inner_join(
    left: Table,
    right: Table,
    binary_predicate: Expression,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Column, Column]: ...
def conditional_left_join(
    left: Table,
    right: Table,
    binary_predicate: Expression,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Column, Column]: ...
def conditional_full_join(
    left: Table,
    right: Table,
    binary_predicate: Expression,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Column, Column]: ...
def conditional_left_semi_join(
    left: Table,
    right: Table,
    binary_predicate: Expression,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def conditional_left_anti_join(
    left: Table,
    right: Table,
    binary_predicate: Expression,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def mixed_inner_join(
    left_keys: Table,
    right_keys: Table,
    left_conditional: Table,
    right_conditional: Table,
    binary_predicate: Expression,
    nulls_equal: NullEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Column, Column]: ...
def mixed_left_join(
    left_keys: Table,
    right_keys: Table,
    left_conditional: Table,
    right_conditional: Table,
    binary_predicate: Expression,
    nulls_equal: NullEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Column, Column]: ...
def mixed_full_join(
    left_keys: Table,
    right_keys: Table,
    left_conditional: Table,
    right_conditional: Table,
    binary_predicate: Expression,
    nulls_equal: NullEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Column, Column]: ...
def mixed_left_semi_join(
    left_keys: Table,
    right_keys: Table,
    left_conditional: Table,
    right_conditional: Table,
    binary_predicate: Expression,
    nulls_equal: NullEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def mixed_left_anti_join(
    left_keys: Table,
    right_keys: Table,
    left_conditional: Table,
    right_conditional: Table,
    binary_predicate: Expression,
    nulls_equal: NullEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...

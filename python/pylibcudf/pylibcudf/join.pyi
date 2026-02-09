# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.expressions import Expression
from pylibcudf.table import Table
from pylibcudf.types import NullEquality

class SetAsBuildTable(IntEnum):
    LEFT = ...
    RIGHT = ...

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

class FilteredJoin:
    def __init__(
        self,
        build: Table,
        compare_nulls: NullEquality,
        reuse_tbl: SetAsBuildTable,
        load_factor: float,
        stream: Stream | None = None,
    ) -> None: ...
    def semi_join(
        self,
        probe: Table,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    def anti_join(
        self,
        probe: Table,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...

# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.types import NanEquality, NanPolicy, NullEquality, NullPolicy

class DuplicateKeepOption(IntEnum):
    KEEP_ANY = ...
    KEEP_FIRST = ...
    KEEP_LAST = ...
    KEEP_NONE = ...

def drop_nulls(
    source_table: Table,
    keys: list[int],
    keep_threshold: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def drop_nans(
    source_table: Table,
    keys: list[int],
    keep_threshold: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def apply_boolean_mask(
    source_table: Table,
    boolean_mask: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def unique(
    input: Table,
    keys: list[int],
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def distinct(
    input: Table,
    keys: list[int],
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
    nans_equal: NanEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def distinct_indices(
    input: Table,
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
    nans_equal: NanEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def stable_distinct(
    input: Table,
    keys: list[int],
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
    nans_equal: NanEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def unique_count(
    source: Column,
    null_handling: NullPolicy,
    nan_handling: NanPolicy,
    stream: Stream | None = None,
) -> int: ...
def distinct_count(
    source: Column,
    null_handling: NullPolicy,
    nan_handling: NanPolicy,
    stream: Stream | None = None,
) -> int: ...

# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum

from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.types import NanEquality, NanPolicy, NullEquality, NullPolicy

class DuplicateKeepOption(IntEnum):
    KEEP_ANY = ...
    KEEP_FIRST = ...
    KEEP_LAST = ...
    KEEP_NONE = ...

def drop_nulls(
    source_table: Table, keys: list[int], keep_threshold: int
) -> Table: ...
def drop_nans(
    source_table: Table, keys: list[int], keep_threshold: int
) -> Table: ...
def apply_boolean_mask(source_table: Table, boolean_mask: Column) -> Table: ...
def unique(
    input: Table,
    keys: list[int],
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
) -> Table: ...
def distinct(
    input: Table,
    keys: list[int],
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
    nans_equal: NanEquality,
) -> Table: ...
def distinct_indices(
    input: Table,
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
    nans_equal: NanEquality,
) -> Column: ...
def stable_distinct(
    input: Table,
    keys: list[int],
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
    nans_equal: NanEquality,
) -> Table: ...
def unique_count(
    source: Column, null_handling: NullPolicy, nan_handling: NanPolicy
) -> int: ...
def distinct_count(
    source: Column, null_handling: NullPolicy, nan_handling: NanPolicy
) -> int: ...

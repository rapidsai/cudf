# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm import DeviceMemoryResource, Stream

from pylibcudf.column import Column
from pylibcudf.copying import OutOfBoundsPolicy
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table
from pylibcudf.types import NanEquality, NullEquality, NullOrder, Order

class ConcatenateNullPolicy(IntEnum):
    IGNORE = ...
    NULLIFY_OUTPUT_ROW = ...

class DuplicateFindOption(IntEnum):
    FIND_FIRST = ...
    FIND_LAST = ...

def explode_outer(
    input: Table,
    explode_column_idx: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def concatenate_rows(
    input: Table,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def concatenate_list_elements(
    input: Column,
    null_policy: ConcatenateNullPolicy,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def contains(
    input: Column,
    search_key: Column | Scalar,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def contains_nulls(
    input: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def index_of(
    input: Column,
    search_key: Column | Scalar,
    find_option: DuplicateFindOption,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def reverse(
    input: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def segmented_gather(
    input: Column,
    gather_map_list: Column,
    bounds_policy: OutOfBoundsPolicy = OutOfBoundsPolicy.DONT_CHECK,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def extract_list_element(
    input: Column,
    index: Column | int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def count_elements(
    input: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def sequences(
    starts: Column,
    sizes: Column,
    steps: Column | None = None,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def sort_lists(
    input: Column,
    sort_order: Order,
    na_position: NullOrder,
    stable: bool = False,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def difference_distinct(
    lhs: Column,
    rhs: Column,
    nulls_equal: NullEquality = NullEquality.EQUAL,
    nans_equal: NanEquality = NanEquality.ALL_EQUAL,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def have_overlap(
    lhs: Column,
    rhs: Column,
    nulls_equal: NullEquality = NullEquality.EQUAL,
    nans_equal: NanEquality = NanEquality.ALL_EQUAL,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def intersect_distinct(
    lhs: Column,
    rhs: Column,
    nulls_equal: NullEquality = NullEquality.EQUAL,
    nans_equal: NanEquality = NanEquality.ALL_EQUAL,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def union_distinct(
    lhs: Column,
    rhs: Column,
    nulls_equal: NullEquality = NullEquality.EQUAL,
    nans_equal: NanEquality = NanEquality.ALL_EQUAL,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def apply_boolean_mask(
    input: Column,
    boolean_mask: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def distinct(
    input: Column,
    nulls_equal: NullEquality,
    nans_equal: NanEquality,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...

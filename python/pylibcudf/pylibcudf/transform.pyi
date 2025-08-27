# Copyright (c) 2024, NVIDIA CORPORATION.
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.expressions import Expression
from pylibcudf.gpumemoryview import gpumemoryview
from pylibcudf.table import Table
from pylibcudf.types import DataType, NullAware

def nans_to_nulls(
    input: Column, stream: Stream | None = None
) -> tuple[gpumemoryview, int]: ...
def compute_column(
    input: Table, expr: Expression, stream: Stream | None = None
) -> Column: ...
def bools_to_mask(
    input: Column, stream: Stream | None = None
) -> tuple[gpumemoryview, int]: ...
def mask_to_bools(
    bitmask: int, begin_bit: int, end_bit: int, stream: Stream | None = None
) -> Column: ...
def transform(
    inputs: list[Column],
    transform_udf: str,
    output_type: DataType,
    is_ptx: bool,
    null_aware: NullAware = NullAware.NO,
    stream: Stream | None = None,
) -> Column: ...
def encode(
    input: Table, stream: Stream | None = None
) -> tuple[Table, Column]: ...
def one_hot_encode(
    input: Column, categories: Column, stream: Stream | None = None
) -> Table: ...

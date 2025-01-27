# Copyright (c) 2024, NVIDIA CORPORATION.
from pylibcudf.column import Column
from pylibcudf.types import DataType, MaskState, TypeId

def make_empty_column(type_or_id: DataType | TypeId) -> Column: ...
def make_numeric_column(
    type_: DataType, size: int, mstate: MaskState
) -> Column: ...
def make_fixed_point_column(
    type_: DataType, size: int, mstate: MaskState
) -> Column: ...
def make_timestamp_column(
    type_: DataType, size: int, mstate: MaskState
) -> Column: ...
def make_duration_column(
    type_: DataType, size: int, mstate: MaskState
) -> Column: ...
def make_fixed_width_column(
    type_: DataType, size: int, mstate: MaskState
) -> Column: ...

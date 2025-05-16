# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table

def fill(
    destination: Column, begin: int, end: int, value: Scalar
) -> Column: ...
def fill_in_place(
    destination: Column, begin: int, end: int, value: Scalar
) -> None: ...
def sequence(size: int, init: Scalar, step: Scalar) -> Column: ...
def repeat(input_table: Table, count: Column | int) -> Table: ...
def calendrical_month_sequence(
    n: int, init: Scalar, months: int
) -> Column: ...

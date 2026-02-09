# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table

def fill(
    destination: Column,
    begin: int,
    end: int,
    value: Scalar,
    stream: Stream | None = None,
) -> Column: ...
def fill_in_place(
    destination: Column,
    begin: int,
    end: int,
    value: Scalar,
    stream: Stream | None = None,
) -> None: ...
def sequence(
    size: int, init: Scalar, step: Scalar, stream: Stream | None = None
) -> Column: ...
def repeat(
    input_table: Table, count: Column | int, stream: Stream | None = None
) -> Table: ...
def calendrical_month_sequence(
    n: int, init: Scalar, months: int, stream: Stream | None = None
) -> Column: ...

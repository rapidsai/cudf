# Copyright (c) 2024, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table

def partition(
    input: Column,
    delimiter: Scalar | None = None,
    stream: Stream | None = None,
) -> Table: ...
def rpartition(
    input: Column,
    delimiter: Scalar | None = None,
    stream: Stream | None = None,
) -> Table: ...

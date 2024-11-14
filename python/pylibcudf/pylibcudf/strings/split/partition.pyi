# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table

def partition(input: Column, delimiter: Scalar | None = None) -> Table: ...
def rpartition(input: Column, delimiter: Scalar | None = None) -> Table: ...

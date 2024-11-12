# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.types import DataType

class Scalar:
    def type(self) -> DataType: ...
    def is_valid(self) -> bool: ...
    @staticmethod
    def empty_like(column: Column) -> Scalar: ...

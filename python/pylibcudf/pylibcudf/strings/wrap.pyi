# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column

def wrap(input: Column, width: int) -> Column: ...

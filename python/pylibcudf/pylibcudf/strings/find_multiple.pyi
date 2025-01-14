# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column

def find_multiple(input: Column, targets: Column) -> Column: ...

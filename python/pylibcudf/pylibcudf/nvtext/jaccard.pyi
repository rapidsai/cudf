# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column

def jaccard_index(input1: Column, input2: Column, width: int) -> Column: ...

# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column

def repeat_strings(input: Column, repeat_times: Column | int) -> Column: ...

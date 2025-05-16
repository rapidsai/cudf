# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def slice_strings(
    input: Column,
    start: Column | Scalar | None = None,
    stop: Column | Scalar | None = None,
    step: Scalar | None = None,
) -> Column: ...

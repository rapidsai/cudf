# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def replace(
    input: Column, target: Scalar, repl: Scalar, maxrepl: int = -1
) -> Column: ...
def replace_multiple(
    input: Column, target: Column, repl: Column, maxrepl: int = -1
) -> Column: ...
def replace_slice(
    input: Column, repl: Scalar | None = None, start: int = 0, stop: int = -1
) -> Column: ...

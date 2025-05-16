# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def replace_tokens(
    input: Column,
    targets: Column,
    replacements: Column,
    delimiter: Scalar | None = None,
) -> Column: ...
def filter_tokens(
    input: Column,
    min_token_length: int,
    replacement: Scalar | None = None,
    delimiter: Scalar | None = None,
) -> Column: ...

# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def ngrams_tokenize(
    input: Column,
    ngrams: int,
    delimiter: Scalar,
    separator: Scalar,
    stream: Stream | None = None,
) -> Column: ...

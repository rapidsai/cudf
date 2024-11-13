# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def ngrams_tokenize(
    input: Column, ngrams: int, delimiter: Scalar, separator: Scalar
) -> Column: ...

# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

class BPEMergePairs:
    def __init__(self, merge_pairs: Column): ...

def byte_pair_encoding(
    input: Column, merge_pairs: BPEMergePairs, separator: Scalar | None = None
) -> Column: ...

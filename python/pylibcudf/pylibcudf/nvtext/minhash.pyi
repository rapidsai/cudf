# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column

def minhash(
    input: Column, seed: int, a: Column, b: Column, width: int
) -> Column: ...
def minhash64(
    input: Column, seed: int, a: Column, b: Column, width: int
) -> Column: ...

# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column import Column

def minhash(
    input: Column, seed: int, a: Column, b: Column, width: int
) -> Column: ...
def minhash64(
    input: Column, seed: int, a: Column, b: Column, width: int
) -> Column: ...
def minhash_ngrams(
    input: Column, ngrams: int, seed: int, a: Column, b: Column
) -> Column: ...
def minhash64_ngrams(
    input: Column, ngrams: int, seed: int, a: Column, b: Column
) -> Column: ...

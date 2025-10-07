# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

def minhash(
    input: Column,
    seed: int,
    a: Column,
    b: Column,
    width: int,
    stream: Stream | None = None,
) -> Column: ...
def minhash64(
    input: Column,
    seed: int,
    a: Column,
    b: Column,
    width: int,
    stream: Stream | None = None,
) -> Column: ...
def minhash_ngrams(
    input: Column,
    ngrams: int,
    seed: int,
    a: Column,
    b: Column,
    stream: Stream | None = None,
) -> Column: ...
def minhash64_ngrams(
    input: Column,
    ngrams: int,
    seed: int,
    a: Column,
    b: Column,
    stream: Stream | None = None,
) -> Column: ...

# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from pylibcudf import nvtext


@acquire_spill_lock()
def minhash(Column strings, Column seeds, int width):
    result = nvtext.minhash.minhash(
        strings.to_pylibcudf(mode="read"),
        seeds.to_pylibcudf(mode="read"),
        width,
    )
    return Column.from_pylibcudf(result)


@acquire_spill_lock()
def minhash64(Column strings, Column seeds, int width):
    result = nvtext.minhash.minhash64(
        strings.to_pylibcudf(mode="read"),
        seeds.to_pylibcudf(mode="read"),
        width,
    )
    return Column.from_pylibcudf(result)


@acquire_spill_lock()
def word_minhash(Column input, Column seeds):
    result = nvtext.minhash.minhash(
        input.to_pylibcudf(mode="read"),
        seeds.to_pylibcudf(mode="read"),
        4,
    )
    return Column.from_pylibcudf(result)


@acquire_spill_lock()
def word_minhash64(Column input, Column seeds):
    result = nvtext.minhash.minhash64(
        input.to_pylibcudf(mode="read"),
        seeds.to_pylibcudf(mode="read"),
        4,
    )
    return Column.from_pylibcudf(result)

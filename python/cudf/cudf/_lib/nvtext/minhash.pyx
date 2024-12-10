# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t, uint64_t

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from pylibcudf import nvtext


@acquire_spill_lock()
def minhash(Column input, uint32_t seed, Column a, Column b, int width):
    return Column.from_pylibcudf(
        nvtext.minhash.minhash(
            input.to_pylibcudf(mode="read"),
            seed,
            a.to_pylibcudf(mode="read"),
            b.to_pylibcudf(mode="read"),
            width,
        )
    )


@acquire_spill_lock()
def minhash64(Column input, uint64_t seed, Column a, Column b, int width):
    return Column.from_pylibcudf(
        nvtext.minhash.minhash64(
            input.to_pylibcudf(mode="read"),
            seed,
            a.to_pylibcudf(mode="read"),
            b.to_pylibcudf(mode="read"),
            width,
        )
    )

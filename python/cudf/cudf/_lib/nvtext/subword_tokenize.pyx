# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t

from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool

from cudf._lib.column cimport Column

from pylibcudf import nvtext


@acquire_spill_lock()
def subword_tokenize_inmem_hash(
    Column strings,
    object hashed_vocabulary,
    uint32_t max_sequence_length=64,
    uint32_t stride=48,
    bool do_lower=True,
    bool do_truncate=False,
):
    """
    Subword tokenizes text series by using the pre-loaded hashed vocabulary
    """
    result = nvtext.subword_tokenize.subword_tokenize(
        strings.to_pylibcudf(mode="read"),
        hashed_vocabulary,
        max_sequence_length,
        stride,
        do_lower,
        do_truncate,
    )
    # return the 3 tensor components
    tokens = Column.from_pylibcudf(result[0])
    masks = Column.from_pylibcudf(result[1])
    metadata = Column.from_pylibcudf(result[2])
    return tokens, masks, metadata

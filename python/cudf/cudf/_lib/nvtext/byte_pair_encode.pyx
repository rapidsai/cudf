# Copyright (c) 2023-2024, NVIDIA CORPORATION.


from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from pylibcudf import nvtext
from pylibcudf.nvtext.byte_pair_encode import BPEMergePairs  # no-cython-lint


@acquire_spill_lock()
def byte_pair_encoding(
    Column strings,
    object merge_pairs,
    object separator
):
    return Column.from_pylibcudf(
        nvtext.byte_pair_encode.byte_pair_encoding(
            strings.to_pylibcudf(mode="read"),
            merge_pairs,
            separator.device_value.c_value
        )
    )

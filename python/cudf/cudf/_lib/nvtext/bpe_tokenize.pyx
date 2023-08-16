# Copyright (c) 2023, NVIDIA CORPORATION.


from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.bpe_tokenize cimport (
    bpe_merge_pairs as cpp_bpe_merge_pairs,
    byte_pair_encoding as cpp_byte_pair_encoding,
    load_merge_pairs_file as cpp_load_merge_pairs_file,
)


cdef class BPE_Merge_Pairs:
    cdef unique_ptr[cpp_bpe_merge_pairs] c_obj

    def __cinit__(self, merges_file):
        cdef string c_merges_file = <string>str(merges_file).encode()
        with nogil:
            self.c_obj = move(cpp_load_merge_pairs_file(c_merges_file))


@acquire_spill_lock()
def byte_pair_encoding(
    Column strings,
    BPE_Merge_Pairs merge_pairs
):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_byte_pair_encoding(c_strings, merge_pairs.c_obj.get()[0]))

    return Column.from_unique_ptr(move(c_result))

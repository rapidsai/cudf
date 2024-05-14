# Copyright (c) 2023-2024, NVIDIA CORPORATION.


from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.nvtext.byte_pair_encode cimport (
    bpe_merge_pairs as cpp_bpe_merge_pairs,
    byte_pair_encoding as cpp_byte_pair_encoding,
    load_merge_pairs as cpp_load_merge_pairs,
)
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.scalar cimport DeviceScalar


cdef class BPEMergePairs:
    cdef unique_ptr[cpp_bpe_merge_pairs] c_obj

    def __cinit__(self, Column merge_pairs):
        cdef column_view c_pairs = merge_pairs.view()
        with nogil:
            self.c_obj = move(cpp_load_merge_pairs(c_pairs))


@acquire_spill_lock()
def byte_pair_encoding(
    Column strings,
    BPEMergePairs merge_pairs,
    object separator
):
    cdef column_view c_strings = strings.view()
    cdef DeviceScalar d_separator = separator.device_value
    cdef const string_scalar* c_separator = <const string_scalar*>d_separator\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_byte_pair_encoding(
                c_strings,
                merge_pairs.c_obj.get()[0],
                c_separator[0]
            )
        )

    return Column.from_unique_ptr(move(c_result))

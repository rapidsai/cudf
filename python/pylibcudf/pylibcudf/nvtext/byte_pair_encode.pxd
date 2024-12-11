# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.byte_pair_encode cimport bpe_merge_pairs
from pylibcudf.scalar cimport Scalar


cdef class BPEMergePairs:
    cdef unique_ptr[bpe_merge_pairs] c_obj

cpdef Column byte_pair_encoding(
    Column input,
    BPEMergePairs merge_pairs,
    Scalar separator=*
)

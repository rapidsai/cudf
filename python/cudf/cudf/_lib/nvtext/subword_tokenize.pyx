# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libc.stdint cimport uint32_t
from libc.stdint cimport uintptr_t

from cudf._lib.move cimport move
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.subword_tokenize cimport (
    subword_tokenize as cpp_subword_tokenize,
    tokenizer_result as cpp_tokenizer_result
)
from cudf._lib.column cimport Column
from cudf import from_dlpack

import numpy as np
import cupy
def device_array_from_ptr(ptr, shape, dtype):
    dtype = np.dtype(dtype)
    elemsize = dtype.itemsize
    datasize = elemsize * shape[0] * shape[1]
    strides = (elemsize*shape[1], elemsize)
    base_mem = cupy.cuda.memory.UnownedMemory(ptr, datasize, None)
    mem = cupy.cuda.MemoryPointer(base_mem, 0)
    return cupy.ndarray(shape, dtype, mem, strides)


def subword_tokenize(Column strings, hash_file, max_sequence_length=64, stride=48, do_lower=True, do_truncate=False, max_num_strings=100, max_num_chars=100000, max_rows_tensor=500):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[cpp_tokenizer_result] c_result

    with nogil:
        c_result = move(
            cpp_subword_tokenize(
                c_strings,
                hash_file.encode(),
                max_sequence_length, stride, do_lower, do_truncate,
                max_num_sentences, max_num_chars, max_rows_tensor
            )
        )

    device_tokenIDS = device_array_from_ptr(<uintptr_t>result.device_tensor_tokenIDS,
                                            shape=(result.nrows_tensor,max_sequence_length),
                                            dtype=np.int32)
    device_mask = device_array_from_ptr(<uintptr_t>result.device_attention_mask,
                                        shape=(result.nrows_tensor,max_sequence_length),
                                        dtype=np.int32)
    device_metadata = device_array_from_ptr(<uintptr_t>result.device_tensor_metadata,
                                            shape=(result.nrows_tensor,3),
                                            dtype=np.int32)

    token = from_dlpack(device_tokenIDS.toDlpack())
    mask = from_dlpack(device_mask.toDlpack())
    metadata = from_dlpack(device_metadata.toDlpack())

    return token, mask, metadata

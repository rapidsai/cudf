# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.libcudf.io.thread cimport (
    set_num_io_threads, num_io_threads
)

cdef cpp_set_num_io_threads(num_io_threads):
    set_num_io_threads(num_io_threads)

cdef cpp_num_io_threads():
    return num_io_threads()

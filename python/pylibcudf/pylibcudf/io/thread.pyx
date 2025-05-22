# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.libcudf.io.thread cimport (
    cpp_set_num_io_threads, cpp_num_io_threads
)

__all__ = [
    "set_num_io_threads",
    "num_io_threads",
]

cpdef void set_num_io_threads(unsigned int num_io_threads):
    cpp_set_num_io_threads(num_io_threads)

cpdef unsigned int num_io_threads():
    return cpp_num_io_threads()

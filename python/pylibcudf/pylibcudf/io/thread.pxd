# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.libcudf.io.thread cimport (
    cpp_set_num_io_threads, cpp_num_io_threads
)

__all__ = [
    "set_num_io_threads",
    "num_io_threads",
]

cpdef set_num_io_threads(num_io_threads)

cpdef num_io_threads()

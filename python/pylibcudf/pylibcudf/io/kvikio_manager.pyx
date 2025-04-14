# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.libcudf.io.kvikio_manager cimport (
    kvikio_manager as cpp_kvikio_manager
)

__all__ = [
    "set_num_io_threads",
    "get_num_io_threads",
    "get_default_num_io_threads",
]

cpdef void set_num_io_threads(unsigned int num_io_threads):
    cpp_kvikio_manager.set_num_io_threads(num_io_threads)

cpdef unsigned int get_num_io_threads():
    return cpp_kvikio_manager.get_num_io_threads()

cpdef unsigned int get_default_num_io_threads():
    return cpp_kvikio_manager.get_default_num_io_threads()

# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.exception_handler cimport libcudf_exception_handler

cdef extern from "cudf/io/thread.hpp" \
        namespace "cudf::io::detail" nogil:

    void set_num_io_threads(unsigned int num_io_threads) \
        except +libcudf_exception_handler

    unsigned int num_io_threads() except +libcudf_exception_handler

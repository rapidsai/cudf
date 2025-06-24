# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.exception_handler cimport libcudf_exception_handler

cdef extern from "cudf/io/thread.hpp" nogil:

    void cpp_set_num_io_threads "cudf::io::detail::set_num_io_threads" \
        (unsigned int num_io_threads) except +libcudf_exception_handler

    unsigned int cpp_num_io_threads "cudf::io::detail::num_io_threads"() \
        except +libcudf_exception_handler

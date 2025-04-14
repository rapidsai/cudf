# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.exception_handler cimport libcudf_exception_handler

cdef extern from "cudf/io/kvikio_manager.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass kvikio_manager:

        @staticmethod
        void set_num_io_threads(unsigned int num_io_threads) \
            except +libcudf_exception_handler

        @staticmethod
        unsigned int get_num_io_threads() except +libcudf_exception_handler

        @staticmethod
        unsigned int get_default_num_io_threads() except +libcudf_exception_handler

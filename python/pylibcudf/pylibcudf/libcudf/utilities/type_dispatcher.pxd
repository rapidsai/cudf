# Copyright (c) 2024, NVIDIA CORPORATION.
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport type_id


cdef extern from "cudf/utilities/type_dispatcher.hpp" namespace "cudf" nogil:
    cdef type_id type_to_id[T]()

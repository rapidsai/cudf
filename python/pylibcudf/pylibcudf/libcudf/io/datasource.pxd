# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from pylibcudf.exception_handler cimport libcudf_exception_handler


cdef extern from "cudf/io/datasource.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass datasource:
        pass

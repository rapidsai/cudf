# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.libcudf.strings.regex_program cimport regex_program


cdef class RegexProgram:
    cdef unique_ptr[regex_program] c_obj

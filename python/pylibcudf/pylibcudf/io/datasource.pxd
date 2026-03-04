# SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.io.datasource cimport datasource


cdef class Datasource:
    cdef datasource* get_datasource(self) except * nogil

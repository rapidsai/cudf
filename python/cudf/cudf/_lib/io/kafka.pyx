# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
from cudf._lib.io.datasource cimport Datasource

cdef class C_KafkaDatasource(Datasource):

    cpdef int create(self):
        print("Inside Kafka.pyx create function!")
        return 1

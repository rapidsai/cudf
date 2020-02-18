# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "kafka_datasource.hpp" namespace "cudf::io::external" nogil:

    bool commit(string topic, int partition, int64_t offset)

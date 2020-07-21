# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.map cimport map
from libc.stdint cimport int64_t
from cudf._lib.io.datasource cimport Datasource
from cudf._lib.cpp.io.types cimport datasource
from libcpp.memory cimport unique_ptr, make_unique


cdef extern from "kafka_consumer.hpp" \
        namespace "cudf::io::external::kafka" nogil:

    cpdef cppclass kafka_consumer:

        kafka_consumer(map[string, string] configs,
                       string topic_name,
                       int partition,
                       int64_t start_offset,
                       int64_t end_offset,
                       int batch_timeout,
                       string delimiter) except +

# cdef class C_KafkaDatasource(Datasource):
cdef unique_ptr[datasource] create(self,
                                   map[string, string] kafka_configs,
                                   string topic,
                                   int partition,
                                   int64_t start_offset,
                                   int64_t end_offset,
                                   int batch_timeout,
                                   string delimiter)

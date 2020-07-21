# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
from cudf._lib.cpp.io.types cimport datasource
from libcpp.memory cimport unique_ptr, make_unique
from cudf_kafka._lib.kafka cimport kafka_consumer

# cdef class C_KafkaDatasource(Datasource):
cdef unique_ptr[datasource] create(self,
                                   map[string, string] kafka_configs,
                                   string topic,
                                   int partition,
                                   int64_t start_offset,
                                   int64_t end_offset,
                                   int batch_timeout,
                                   string delimiter):
    print("Creating c_datasource in kafka.pyx create()")
    return <unique_ptr[datasource]> make_unique[kafka_consumer](kafka_configs,
                                                                topic,
                                                                partition,
                                                                start_offset,
                                                                end_offset,
                                                                batch_timeout,
                                                                delimiter)

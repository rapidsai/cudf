# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.map cimport map
from libc.stdint cimport int64_t
from cudf._lib.cpp.io.types cimport datasource
from libcpp.memory cimport unique_ptr, make_unique
from cudf_kafka._lib.kafka cimport kafka_consumer

cdef class KafkaDatasource(Datasource):

    def __init__(
        self,
        kafka_configs,
        topic,
        partition,
        start_offset,
        end_offset,
        batch_timeout,
        delimiter,
    ):

        self.kafka_configs = kafka_configs
        self.topic = topic
        self.partition = partition
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.batch_timeout = batch_timeout
        self.delimiter = delimiter

    def __cinit__(self,
                  map[string, string] kafka_configs,
                  string topic,
                  int partition,
                  int64_t start_offset,
                  int64_t end_offset,
                  int batch_timeout,
                  string delimiter,):
        self.c_datasource = <unique_ptr[datasource]> \
            make_unique[kafka_consumer](kafka_configs,
                                        topic,
                                        partition,
                                        start_offset,
                                        end_offset,
                                        batch_timeout,
                                        delimiter)

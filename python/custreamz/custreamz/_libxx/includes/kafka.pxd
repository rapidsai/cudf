# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool
from libc.stdint cimport int32_t, int64_t


cdef extern from "kafka_datasource.hpp" namespace "cudf::io::external" nogil:

    cdef cppclass kafka_datasource:

        kafka_datasource(map[string, string] configs,
                         vector[string] topics,
                         vector[int] partitions) except +

        string libcudf_datasource_identifier() except +

        void print_consumer_metadata() except +

        map[string, string] current_configs() except +

        bool commit_offset(string topic, int partition, int offset) except +

        map[int, int64_t] get_committed_offset(string topic,
                                               vector[int] partitions) except +

        map[string, int64_t] get_watermark_offset(string topic,
                                                  int32_t partition) except +

        string consume_range(string topic,
                             int partition,
                             int64_t start_offset,
                             int64_t end_offset,
                             int batch_timeout,
                             string delimiter) except +

        bool produce_message(string topic,
                             string message_val,
                             string message_key) except +

        bool flush(int timeout) except +

        bool unsubscribe() except +

        bool close() except +

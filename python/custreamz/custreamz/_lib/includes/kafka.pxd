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


cdef extern from "kafka_consumer.hpp" namespace "cudf::io::external" nogil:

    cpdef cppclass kafka_consumer:

        kafka_consumer(map[string, string] configs) except +

        bool assign(vector[string] topics, vector[int] partitions) except +

        string libcudf_datasource_identifier() except +

        void print_consumer_metadata() except +

        map[string, string] current_configs() except +

        bool commit_offset(string topic, int partition, int offset) except +

        int64_t get_committed_offset(string topic,
                                     int partition) except +

        map[string, int64_t] get_watermark_offset(string topic,
                                                  int partition,
                                                  int timeout,
                                                  bool cached) except +

        string consume_range(string topic,
                             int partition,
                             int64_t start_offset,
                             int64_t end_offset,
                             int batch_timeout,
                             string delimiter) except +

        bool unsubscribe() except +

        bool close(int timeout) except +


cdef extern from "kafka_producer.hpp" namespace "cudf::io::external" nogil:

    cpdef cppclass kafka_producer:

        kafka_producer(map[string, string] configs) except +

        string libcudf_datasource_identifier() except +

        map[string, string] current_configs() except +

        bool produce_message(string topic,
                             string message_val,
                             string message_key) except +

        bool flush(int timeout) except +

        bool close(int timeout) except +


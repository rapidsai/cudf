# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from cudf._lib.io.datasource cimport Datasource
from cudf._lib.pylibcudf.libcudf.io.datasource cimport datasource


cdef extern from "cudf_kafka/kafka_callback.hpp" \
        namespace "cudf::io::external::kafka" nogil:
    ctypedef object (*python_callable_type)()


cdef extern from "cudf_kafka/kafka_consumer.hpp" \
        namespace "cudf::io::external::kafka" nogil:

    cpdef cppclass kafka_consumer:

        kafka_consumer(map[string, string] configs,
                       python_callable_type python_callable) except +

        kafka_consumer(map[string, string] configs,
                       python_callable_type python_callable,
                       string topic_name,
                       int32_t partition,
                       int64_t start_offset,
                       int64_t end_offset,
                       int32_t batch_timeout,
                       string delimiter) except +

        bool assign(vector[string] topics, vector[int32_t] partitions) except +

        void commit_offset(string topic,
                           int32_t partition,
                           int64_t offset) except +

        int64_t get_committed_offset(string topic,
                                     int32_t partition) except +

        map[string, vector[int32_t]] list_topics(string topic) except +

        map[string, int64_t] get_watermark_offset(string topic,
                                                  int32_t partition,
                                                  int32_t timeout,
                                                  bool cached) except +

        void unsubscribe() except +

        void close(int32_t timeout) except +

cdef class KafkaDatasource(Datasource):

    cdef unique_ptr[datasource] c_datasource
    cdef string topic
    cdef int32_t partition
    cdef int64_t start_offset
    cdef int64_t end_offset
    cdef int32_t batch_timeout
    cdef string delimiter

    cdef datasource* get_datasource(self) nogil

    cpdef void commit_offset(self,
                             string topic,
                             int32_t partition,
                             int64_t offset)

    cpdef int64_t get_committed_offset(self, string topic, int32_t partition)

    cpdef map[string, vector[int32_t]] list_topics(self, string tp) except *

    cpdef map[string, int64_t] get_watermark_offset(self, string topic,
                                                    int32_t partition,
                                                    int32_t timeout,
                                                    bool cached)

    cpdef void unsubscribe(self)

    cpdef void close(self, int32_t timeout)

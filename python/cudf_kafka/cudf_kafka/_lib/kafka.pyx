# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.map cimport map
from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool
from cudf._lib.cpp.io.types cimport datasource
from libcpp.memory cimport unique_ptr, make_unique
from cudf_kafka._lib.kafka cimport kafka_consumer

cdef class KafkaDatasource(Datasource):

    def __cinit__(self,
                  map[string, string] kafka_configs,
                  string topic=b"",
                  int32_t partition=-1,
                  int64_t start_offset=0,
                  int64_t end_offset=0,
                  int32_t batch_timeout=10000,
                  string delimiter=b"",):
        if topic != b"" and partition != -1:
            self.c_datasource = <unique_ptr[datasource]> \
                make_unique[kafka_consumer](kafka_configs,
                                            topic,
                                            partition,
                                            start_offset,
                                            end_offset,
                                            batch_timeout,
                                            delimiter)
        else:
            self.c_datasource = <unique_ptr[datasource]> \
                make_unique[kafka_consumer](kafka_configs)

    cdef datasource* get_datasource(self) nogil:
        return <datasource *> self.c_datasource.get()

    cpdef void commit_offset(self,
                             string topic,
                             int32_t partition,
                             int64_t offset):
        (<kafka_consumer *> self.c_datasource.get()).commit_offset(
            topic, partition, offset)

    cpdef int64_t get_committed_offset(self,
                                       string topic,
                                       int32_t partition):
        return (<kafka_consumer *> self.c_datasource.get()). \
            get_committed_offset(topic, partition)

    cpdef map[string, vector[int32_t]] list_topics(self,
                                                   string topic) except *:
        return (<kafka_consumer *> self.c_datasource.get()). \
            list_topics(topic)

    cpdef map[string, int64_t] get_watermark_offset(self, string topic,
                                                    int32_t partition,
                                                    int32_t timeout,
                                                    bool cached):
        return (<kafka_consumer *> self.c_datasource.get()). \
            get_watermark_offset(topic, partition, timeout, cached)

    cpdef void unsubscribe(self):
        (<kafka_consumer *> self.c_datasource.get()).unsubscribe()

    cpdef void close(self, int32_t timeout):
        (<kafka_consumer *> self.c_datasource.get()).close(timeout)

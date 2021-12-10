# Copyright (c) 2020, NVIDIA CORPORATION.

cimport cpython
from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.io.types cimport datasource

from cudf_kafka._lib.kafka cimport kafka_consumer

import functools


cdef map[string, string] oauth_callback_wrapper(void *ctx):
    print("Entering oauth_callback_wrapper")

    # ctx is a functools.partial
    func, args = <object>(ctx)

    # Never makes it here, ^^
    print("Func: " + str(func))
    print("Args: " + str(args))
    ret = func(*args)
    return ret


cdef class KafkaDatasource(Datasource):

    def __cinit__(self,
                  object kafka_configs,
                  string topic=b"",
                  int32_t partition=-1,
                  int64_t start_offset=0,
                  int64_t end_offset=0,
                  int32_t batch_timeout=10000,
                  string delimiter=b"",):

        cdef map[string, string] configs
        cdef void* python_callable
        cdef map[string, string] (*cb)(void *)

        for key in kafka_configs:
            if key == 'oauth_cb':
                if callable(kafka_configs[key]):
                    python_callable = <void *>kafka_configs[key]
                    cb = &oauth_callback_wrapper
                else:
                    raise TypeError("'oauth_cb' configuration must \
                                      be a Python callable object")
            else:
                configs[key.encode()] = kafka_configs[key].encode()

        if topic != b"" and partition != -1:
            self.c_datasource = <unique_ptr[datasource]> \
                make_unique[kafka_consumer](configs,
                                            python_callable,
                                            cb,
                                            topic,
                                            partition,
                                            start_offset,
                                            end_offset,
                                            batch_timeout,
                                            delimiter)
        else:
            self.c_datasource = <unique_ptr[datasource]> \
                make_unique[kafka_consumer](configs, python_callable, cb)

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

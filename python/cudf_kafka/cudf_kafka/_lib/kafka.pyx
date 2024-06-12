# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool, nullptr
from libcpp.map cimport map
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf.io.datasource cimport datasource

from cudf_kafka._lib.kafka cimport kafka_consumer


# To avoid including <python.h> in libcudf_kafka
# we introduce this wrapper in Cython
cdef map[string, string] oauth_callback_wrapper(void *ctx):
    resp = (<object>(ctx))()
    cdef map[string, string] c_resp
    c_resp[str.encode("token")] = str.encode(resp["token"])
    c_resp[str.encode("token_expiration_in_epoch")] \
        = str(resp["token_expiration_in_epoch"]).encode()
    return c_resp


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
        cdef void* python_callable = nullptr
        cdef map[string, string] (*python_callable_wrapper)(void *)

        for key in kafka_configs:
            if key == 'oauth_cb':
                if callable(kafka_configs[key]):
                    python_callable = <void *>kafka_configs[key]
                    python_callable_wrapper = &oauth_callback_wrapper
                else:
                    raise TypeError("'oauth_cb' configuration must \
                                      be a Python callable object")
            else:
                configs[key.encode()] = kafka_configs[key].encode()

        if topic != b"" and partition != -1:
            self.c_datasource = <unique_ptr[datasource]> \
                move(make_unique[kafka_consumer](configs,
                                                 python_callable,
                                                 python_callable_wrapper,
                                                 topic,
                                                 partition,
                                                 start_offset,
                                                 end_offset,
                                                 batch_timeout,
                                                 delimiter))
        else:
            self.c_datasource = <unique_ptr[datasource]> \
                move(make_unique[kafka_consumer](configs,
                                                 python_callable,
                                                 python_callable_wrapper))

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

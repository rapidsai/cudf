# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *
from custreamz._libxx.includes.kafka cimport (
    kafka_datasource as kafka_external,
)

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libcpp.map cimport map
from cython.operator cimport dereference, postincrement
from libc.stdint cimport uint32_t, int64_t

# Global Kafka configurations
cdef kafka_external *kds
cdef string ds_id

cpdef create_kafka_handle(kafka_conf):
    global kds, ds_id
    cdef map[string, string] kafka_confs
    for key, value in kafka_conf.items():
        kafka_confs[str.encode(key)] = str.encode(value)
    kds = new kafka_external(kafka_confs)
    ds_id = kds.libcudf_datasource_identifier()


cpdef get_watermark_offsets(datasource_id=None,
                            topic=None,
                            partition=1):

    topic = "libcudf-test"

    cdef map[string, int64_t] offsets = \
        kds.get_watermark_offset(str.encode(topic), partition)
    cdef map[string, int64_t].iterator it = offsets.begin()

    while(it != offsets.end()):
        print("Topic: " + str(dereference(it).first) +
              " Offset: " + str(dereference(it).second))
        postincrement(it)

cpdef read_gdf(data_format="blob",
               lines=True,
               start=0,
               end=1000,
               timeout=10000):

    if data_format == 'blob':
        json_str = kds.consume_range(start,
                                     end,
                                     timeout)
        print("JSON -> " + str(json_str))
    else:
        print("Need to do something else here .... like use libcudf ....")

    return json_str


cpdef commit_offsets():
    cdef kafka_external *kds = new kafka_external()
    cdef string ds_id = kds.libcudf_datasource_identifier()
    print("Kafka Datasource ID: " + str(ds_id))

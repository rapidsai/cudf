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


cpdef get_watermark_offsets(datasource_id=None,
                            kafka_configs=None,
                            topic=None,
                            partition=-1):
    print("before")
    cdef kafka_external *kds = new kafka_external()
    cdef string ds_id = kds.libcudf_datasource_identifier()
    print("Kafka Datasource ID: " + str(ds_id))
    cdef map[string, int64_t] offsets = \
        kds.get_watermark_offset(topic, partition)
    print("After getting watermarks")
    cdef map[string, int64_t].iterator it = offsets.begin()

    print("Before loop")
    while(it != offsets.end()):
        print("inside loop")
        # let's pretend here I just want to print the key and the value
        print(dereference(it).first)  # print the key
        print(dereference(it).second)  # print the associated value
        postincrement(it)  # Increment the iterator to the net element

cpdef read_json_example():
    print("Reading JSON .....")

cpdef commit_offsets():
    cdef kafka_external *kds = new kafka_external()
    cdef string ds_id = kds.libcudf_datasource_identifier()
    print("Kafka Datasource ID: " + str(ds_id))

# Copyright (c) 2019-2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from custreamz._libxx.includes.kafka cimport (
    kafka_datasource as kafka_external,
)
from cudf._libxx.table cimport Table
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.move cimport move

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.map cimport map
from cython.operator cimport dereference, postincrement
from libc.stdint cimport uint32_t, int64_t
cimport cudf._libxx.cpp.io.functions as libcudf
cimport cudf._libxx.cpp.io.types as cudf_io_types

# Global Kafka configurations
cdef kafka_external *kds
cdef string ds_id

cpdef create_kafka_consumer(kafka_conf):
    global kds, ds_id
    cdef map[string, string] kafka_confs
    for key, value in kafka_conf.items():
        kafka_confs[str.encode(key)] = str.encode(value)
    cdef vector[string] v_topics
    kds = new kafka_external(kafka_confs)
    ds_id = kds.libcudf_datasource_identifier()


cpdef assign(topics=[], partitions=[]):

    cdef vector[string] v_topics
    cdef vector[int] v_partitions

    for top in topics:
        v_topics.push_back(top.encode())
    for part in partitions:
        v_partitions.push_back(part)

    return kds.assign(v_topics, v_partitions)

cpdef read_gdf(lines=True,
               dtype=True,
               compression="infer",
               dayfirst=True,
               byte_range=None,
               topic=None,
               partition=0,
               start=-1,
               end=-1,
               timeout=10000,
               delimiter="\n"):

    cdef json_str = kds.consume_range(str.encode(topic),
                                      partition,
                                      start,
                                      end,
                                      timeout,
                                      str.encode(delimiter))

    cdef cudf_io_types.table_with_metadata c_out_table
    cdef libcudf.read_json_args json_args = libcudf.read_json_args()

    if len(json_str) > 0:
        json_args.lines = lines
        json_args.source = cudf_io_types.source_info(
            json_str, len(json_str))
        json_args.compression = cudf_io_types.compression_type.NONE
        json_args.dayfirst = dayfirst

        with nogil:
            c_out_table = move(libcudf.read_json(json_args))

        column_names = [x.decode() for x in c_out_table.metadata.column_names]
        return Table.from_unique_ptr(move(c_out_table.tbl),
                                     column_names=column_names)
    else:
        return None

cpdef get_committed_offset(topic=None, partition=None):
    return kds.get_committed_offset(str.encode(topic), partition)

cpdef current_configs():
    kds.current_configs()

cpdef print_consumer_metadata():
    kds.print_consumer_metadata()

cpdef get_watermark_offsets(topic=None,
                            partition=0,
                            timeout=10000,
                            cached=False):

    cdef map[string, int64_t] offsets = \
        kds.get_watermark_offset(str.encode(topic), partition, timeout, cached)
    return offsets

cpdef commit_topic_offset(topic=None,
                          partition=0,
                          offset=0,
                          asynchronous=True):
    return kds.commit_offset(str.encode(topic), partition, offset)

cpdef produce_message(topic=None, message_val=None, message_key=None):
    return kds.produce_message(str.encode(topic),
                               str.encode(message_val),
                               str.encode(message_key))

cpdef flush(timeout=10000):
    return kds.flush(timeout)

cpdef unsubscribe():
    return kds.unsubscribe()

cpdef close():
    return kds.close()

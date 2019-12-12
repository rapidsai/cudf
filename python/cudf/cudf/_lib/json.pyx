# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._lib.includes.json cimport (
    Conf as KafkaConf,
    reader as json_reader,
    reader_options as json_reader_options
)
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr

from cudf._lib.utils cimport *
from cudf._lib.utils import *

import collections.abc as abc
import os


def is_file_like(obj):
    if not (hasattr(obj, 'read') or hasattr(obj, 'write')):
        return False
    if not hasattr(obj, "__iter__"):
        return False
    return True


cpdef read_json(filepath_or_buffer, dtype, lines, compression, byte_range, kafka_configs):
    """
    Cython function to call into libcudf API, see `read_json`.

    See Also
    --------
    cudf.io.json.read_json
    cudf.io.json.to_json
    """

    # Setup arguments
    print("Before args")
    cdef json_reader_options args = json_reader_options()
    args.lines = lines
    if compression is not None:
        args.compression = compression.encode()
    if dtype is False:
        raise ValueError("False value is unsupported for `dtype`")
    elif dtype is not True:
        if isinstance(dtype, abc.Mapping):
            for k, v in dtype.items():
                args.dtype.push_back(str(str(k) + ":" + str(v)).encode())
        elif not isinstance(dtype, abc.Iterable):
            raise TypeError("`dtype` must be 'list like' or 'dict'")
        else:
            for col_dtype in dtype:
                args.dtype.push_back(str(col_dtype).encode())

    print("After args")
    # Create reader from source
    cdef const unsigned char[::1] buffer
    cdef string filepath

    # Kafka execution path variables
    print("Before Kafka configuration")
    cdef KafkaConf.ConfType gk = KafkaConf.ConfType.CONF_GLOBAL
    cdef KafkaConf *gkc
    cdef vector[string] topics

    if kafka_configs is not None:
        print("Making kafka configurations")
        errstr=""
        gkc = KafkaConf.create(KafkaConf.ConfType.CONF_GLOBAL)

        for kafka_conf_key in kafka_configs:
            if kafka_conf_key == 'cudf.topic.list':
                tops = kafka_configs[kafka_conf_key].split(',')
                for t in tops:
                    topics.push_back(t.encode())
            elif kafka_conf_key == 'cudf.start.offset':
                args.cudf_start_offset = kafka_configs[kafka_conf_key]
            elif kafka_conf_key == 'cudf.end.offset':
                args.cudf_end_offset = kafka_configs[kafka_conf_key]
            else:
                gkc.set(kafka_conf_key.encode(), kafka_configs[kafka_conf_key].encode(), errstr.encode())
        print("End kafka configuration")
    else:
        print("Taking file path approach")
        buffer = view_of_buffer(filepath_or_buffer)
        if buffer is None:
            if os.path.isfile(filepath_or_buffer):
                filepath = <string>str(filepath_or_buffer).encode()
            else:
                buffer = filepath_or_buffer.encode()

    cdef unique_ptr[json_reader] reader
    print("Right here")
    with nogil:
        if kafka_configs is not None:
            reader = unique_ptr[json_reader](
                new json_reader(gkc, topics, args)
            )
        elif buffer is None:
            reader = unique_ptr[json_reader](
                new json_reader(filepath, args)
            )
        else:
            reader = unique_ptr[json_reader](
                new json_reader(<char *>&buffer[0], buffer.shape[0], args)
            )

    # Read data into columns
    cdef cudf_table c_out_table
    cdef size_t c_range_offset = byte_range[0] if byte_range is not None else 0
    cdef size_t c_range_size = byte_range[1] if byte_range is not None else 0
    with nogil:
        if c_range_offset !=0 or c_range_size != 0:
            c_out_table = reader.get().read_byte_range(
                c_range_offset, c_range_size
            )
        else:
            c_out_table = reader.get().read()

    return table_to_dataframe(&c_out_table)

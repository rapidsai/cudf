# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "cudf/cudf.h" namespace "RdKafka" nogil:

    cdef cppclass Conf:

        enum ConfType:
            CONF_GLOBAL "RdKafka::Conf::CONF_GLOBAL",
            CONF_TOPIC "RdKafka::Conf::CONF_TOPIC",

        enum ConfResult:
            CONF_UNKNOWN = -2,
            CONF_INVALID = -1,
            CONF_OK = 0,

        @staticmethod
        Conf *create(ConfType) except +

        ConfResult set(string &name, string &value, string &errstr) except +

cdef extern from "cudf/cudf.h" namespace "cudf::io::json" nogil:

    cdef cppclass reader_options:
        bool lines
        string compression
        vector[string] dtype
        Conf *kafka_conf
        vector[string] topics
        int cudf_start_offset
        int cudf_end_offset

        reader_options() except +

        reader_options(
            bool lines,
            string compression,
            vector[string] dtype
        ) except +

    cdef cppclass reader:

        reader(
            Conf *global_configs,
            vector[string] topics,
            const reader_options &options
        ) except +

        reader(
            string filepath,
            const reader_options &args
        ) except +

        reader(
            const char *buffer,
            size_t length,
            const reader_options &args
        ) except +

        cudf_table read() except +

        cudf_table read_byte_range(size_t offset, size_t size) except +

# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.io.types cimport TableWithMetadata
from cudf._lib.pylibcudf.libcudf.io.avro cimport avro_reader_options


cdef class AvroReaderOptions:
    cdef avro_reader_options avro_opts

cpdef TableWithMetadata read_avro(AvroReaderOptions options)

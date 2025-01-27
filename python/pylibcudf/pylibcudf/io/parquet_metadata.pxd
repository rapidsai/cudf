# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.io.types cimport SourceInfo
from pylibcudf.libcudf.io.parquet_metadata cimport(
    parquet_metadata,
    parquet_schema,
    parquet_column_schema,
)

cdef class ParquetColumnSchema:
    cdef parquet_column_schema column_schema

    @staticmethod
    cdef from_column_schema(parquet_column_schema column_schema)

    cpdef str name(self)

    cpdef int num_children(self)

    cpdef ParquetColumnSchema child(self, int idx)

    cpdef list children(self)


cdef class ParquetSchema:
    cdef parquet_schema schema

    @staticmethod
    cdef from_schema(parquet_schema schema)

    cpdef ParquetColumnSchema root(self)


cdef class ParquetMetadata:
    cdef parquet_metadata meta

    @staticmethod
    cdef from_metadata(parquet_metadata meta)

    cpdef ParquetSchema schema(self)

    cpdef int num_rows(self)

    cpdef int num_rowgroups(self)

    cpdef dict metadata(self)

    cpdef list rowgroup_metadata(self)


cpdef ParquetMetadata read_parquet_metadata(SourceInfo src_info)

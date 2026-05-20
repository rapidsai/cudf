# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.io.types cimport SourceInfo
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData as cpp_FileMetaData
from pylibcudf.libcudf.io.parquet_metadata cimport(
    parquet_metadata,
    parquet_schema,
    parquet_column_schema,
)
from pylibcudf.types cimport DataType

cdef class ParquetColumnSchema:
    cdef parquet_column_schema column_schema

    @staticmethod
    cdef from_column_schema(parquet_column_schema column_schema)

    cpdef str name(self)

    cpdef int num_children(self)

    cpdef ParquetColumnSchema child(self, int idx)

    cpdef list children(self)

    cpdef DataType cudf_type(self)


cdef class ParquetSchema:
    cdef parquet_schema schema

    @staticmethod
    cdef from_schema(parquet_schema schema)

    cpdef ParquetColumnSchema root(self)

    cpdef dict column_types(self)


cdef class ParquetMetadata:
    cdef parquet_metadata meta

    @staticmethod
    cdef from_metadata(parquet_metadata meta)

    cpdef ParquetSchema schema(self)

    cpdef int num_rows(self)

    cpdef int num_rowgroups(self)

    cpdef list num_rowgroups_per_file(self)

    cpdef dict metadata(self)

    cpdef list rowgroup_metadata(self)

    cpdef dict columnchunk_metadata(self)

cdef class FileMetaData:
    cdef cpp_FileMetaData c_obj

    @staticmethod
    cdef FileMetaData from_cpp(cpp_FileMetaData metadata)

cpdef ParquetMetadata read_parquet_metadata(SourceInfo src_info)
cpdef list read_parquet_footers(SourceInfo src_info)

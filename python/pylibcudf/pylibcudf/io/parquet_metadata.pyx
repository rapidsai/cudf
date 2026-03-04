# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.io.types cimport SourceInfo
from pylibcudf.libcudf.io cimport parquet_metadata as cpp_parquet_metadata
from pylibcudf.types cimport DataType


__all__ = [
    "ParquetColumnSchema",
    "ParquetMetadata",
    "ParquetSchema",
    "read_parquet_metadata",
]

cdef class ParquetColumnSchema:
    """
    Schema of a parquet column, including the nested columns.

    Parameters
    ----------
    parquet_column_schema
    """
    def __init__(self):
        raise ValueError("Construct ParquetColumnSchema with from_column_schema.")

    @staticmethod
    cdef from_column_schema(cpp_parquet_metadata.parquet_column_schema column_schema):
        cdef ParquetColumnSchema result = ParquetColumnSchema.__new__(
            ParquetColumnSchema
        )
        result.column_schema = column_schema
        return result

    cpdef str name(self):
        """
        Returns parquet column name; can be empty.

        Returns
        -------
        str
            Column name
        """
        return self.column_schema.name().decode()

    cpdef int num_children(self):
        """
        Returns the number of child columns.

        Returns
        -------
        int
            Children count
        """
        return self.column_schema.num_children()

    cpdef ParquetColumnSchema child(self, int idx):
        """
        Returns schema of the child with the given index.

        Parameters
        ----------
        idx : int
            Child Index

        Returns
        -------
        ParquetColumnSchema
            Child schema
        """
        return ParquetColumnSchema.from_column_schema(self.column_schema.child(idx))

    cpdef list children(self):
        """
        Returns schemas of all child columns.

        Returns
        -------
        list[ParquetColumnSchema]
            Child schemas.
        """
        cdef cpp_parquet_metadata.parquet_column_schema child
        return [
            ParquetColumnSchema.from_column_schema(child)
            for child in self.column_schema.children()
        ]

    cpdef DataType cudf_type(self):
        """
        Returns the cudf data type for this column.

        This is the resolved cudf data type mapped from the parquet
        physical/logical types.

        Returns
        -------
        DataType
            cudf data type
        """
        return DataType.from_libcudf(self.column_schema.cudf_type())


cdef class ParquetSchema:
    """
    Schema of a parquet file.

    Parameters
    ----------
    parquet_schema
    """

    def __init__(self):
        raise ValueError("Construct ParquetSchema with from_schema.")

    @staticmethod
    cdef from_schema(cpp_parquet_metadata.parquet_schema schema):
        cdef ParquetSchema result = ParquetSchema.__new__(ParquetSchema)
        result.schema = schema
        return result

    cpdef ParquetColumnSchema root(self):
        """
        Returns the schema of the struct column that contains all columns as fields.

        Returns
        -------
        ParquetColumnSchema
            Root column schema
        """
        return ParquetColumnSchema.from_column_schema(self.schema.root())

    cpdef dict column_types(self):
        """
        Returns a dictionary mapping column names to their cudf data types.

        Returns
        -------
        dict[str, DataType]
            Dictionary mapping column names to DataType objects
        """
        cdef ParquetColumnSchema root_schema = self.root()
        return {
            root_schema.child(i).name(): root_schema.child(i).cudf_type()
            for i in range(root_schema.num_children())
        }


cdef class ParquetMetadata:
    """
    Information about content of a parquet file.

    Parameters
    ----------
    parquet_metadata
    """

    def __init__(self):
        raise ValueError("Construct ParquetMetadata with from_metadata.")

    @staticmethod
    cdef from_metadata(cpp_parquet_metadata.parquet_metadata meta):
        cdef ParquetMetadata result = ParquetMetadata.__new__(ParquetMetadata)
        result.meta = meta
        return result

    cpdef ParquetSchema schema(self):
        """
        Returns the parquet schema.

        Returns
        -------
        ParquetSchema
            Parquet schema
        """
        return ParquetSchema.from_schema(self.meta.schema())

    cpdef int num_rows(self):
        """
        Returns the number of rows of the root column.

        Returns
        -------
        int
            Number of rows
        """
        return self.meta.num_rows()

    cpdef int num_rowgroups(self):
        """
        Returns the total number of rowgroups in the file.

        Returns
        -------
        int
            Number of row groups.
        """
        return self.meta.num_rowgroups()

    cpdef list num_rowgroups_per_file(self):
        """
        Returns the number of rowgroups in each file.
        """
        return self.meta.num_rowgroups_per_file()

    cpdef dict metadata(self):
        """
        Returns the key-value metadata in the file footer.

        Returns
        -------
        dict[str, str]
            Key value metadata as a map.
        """
        return {key.decode(): val.decode() for key, val in self.meta.metadata()}

    cpdef list rowgroup_metadata(self):
        """
        Returns the row group metadata in the file footer.

        Returns
        -------
        list[dict[str, int]]
            Vector of row group metadata as maps.
        """
        return [
            {key.decode(): val for key, val in metadata}
            for metadata in self.meta.rowgroup_metadata()
        ]

    cpdef dict columnchunk_metadata(self):
        """
        Returns a map of leaf column names to lists of `total_uncompressed_size`
        metadata from all column chunks in the file footer.

        Returns
        -------
        dict[str, list[int]]
            Map of leaf column names to lists of `total_uncompressed_size` metadata
            from all their column chunks.
        """
        return {
            col_name.decode(): uncompressed_sizes
            for col_name, uncompressed_sizes in self.meta.columnchunk_metadata()
        }


cpdef ParquetMetadata read_parquet_metadata(SourceInfo src_info):
    """
    Reads metadata of parquet dataset.

    Parameters
    ----------
    src_info : SourceInfo
        Dataset source.

    Returns
    -------
    ParquetMetadata
        Parquet_metadata with parquet schema, number of rows,
        number of row groups and key-value metadata.
    """
    cdef cpp_parquet_metadata.parquet_metadata c_result

    with nogil:
        c_result = cpp_parquet_metadata.read_parquet_metadata(src_info.c_obj)

    return ParquetMetadata.from_metadata(c_result)

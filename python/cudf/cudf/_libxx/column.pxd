from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from cudf._libxx.lib cimport *

cimport cudf._lib.cudf as gdf

cdef class Column:
    cdef dict __dict__    
    cdef column_view view(self) except *
    cdef mutable_column_view mutable_view(self) except *

    @staticmethod
    cdef Column from_ptr(unique_ptr[column] c_col)

    cdef size_type null_count(self)

    cdef gdf.gdf_column* gdf_column_view(self) except *

    @staticmethod
    cdef Column from_gdf_column(gdf.gdf_column* c_col)

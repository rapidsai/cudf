# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .cudf_cpp cimport *
from .cudf_cpp import *

from libc.stdlib cimport free


def apply_cast(incol, outcol):
    """
      Cast from incol.dtype to outcol.dtype
    """

    check_gdf_compatibility(incol)
    check_gdf_compatibility(outcol)
    
    cdef gdf_column* c_incol = column_view_from_column(incol)
    cdef gdf_column* c_outcol = column_view_from_column(outcol)

    cdef gdf_error result
    with nogil:    
        result = gdf_cast(
            <gdf_column*>c_incol,
            <gdf_column*>c_outcol)
    
    free(c_incol)
    free(c_outcol)

    check_gdf_error(result)


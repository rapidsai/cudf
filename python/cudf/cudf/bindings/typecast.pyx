# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .cudf_cpp cimport *
from .cudf_cpp import *
from cudf.bindings.unaryops cimport *
from cudf.dataframe.column import Column
from libc.stdlib cimport free

import numpy as np

_time_unit = {
    'none'  : TIME_UNIT_NONE,
    's'     : TIME_UNIT_s,
    'ms'    : TIME_UNIT_ms,
    'us'    : TIME_UNIT_us,
    'ns'    : TIME_UNIT_ns,
}

def apply_cast(incol, **kwargs):
    """
      Cast from incol.dtype to outcol.dtype
    """

    check_gdf_compatibility(incol)
    check_gdf_compatibility(outcol)

    cdef gdf_column* c_incol = column_view_from_column(incol)

    npdtype = kwargs.get("dtype", np.float64)
    cdef gdf_dtype dtype = dtypes[npdtype]
    cdef uintptr_t category

    cdef gdf_dtype_extra_info info = gdf_dtype_extra_info(
        time_unit = TIME_UNIT_NONE,
        category = <void*>category
    )
    unit = kwargs.get("time_unit", 'none')
    info.time_unit = _time_unit[unit]

    cdef gdf_column result

    with nogil:    
        result = col_cast(
          c_incol[0],
          dtype,
          info
       )
    
    free(c_incol)
    data, mask = gdf_column_to_column_mem(&result)
    return Column.from_mem_views(data, mask)

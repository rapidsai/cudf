# Copyright (c) 2019-2020, NVIDIA CORPORATION.

# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *

cdef extern from "cudf/legacy/search.hpp" namespace "cudf" nogil:

    cdef gdf_column lower_bound(
        const cudf_table& t,
        const cudf_table& values,
        vector[bool] c_desc_flags,
    ) except +

    cdef gdf_column upper_bound(
        const cudf_table& t,
        const cudf_table& values,
        vector[bool] c_desc_flags,
    ) except +

    cdef bool contains(
        const gdf_column& t,
        const gdf_scalar& value
    ) except +

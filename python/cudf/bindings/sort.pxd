# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf.h" nogil:

    ctypedef struct _OpaqueRadixsortPlan:
        pass
    ctypedef struct  gdf_radixsort_plan_type:
        pass

    ctypedef struct _OpaqueSegmentedRadixsortPlan:
        pass
    ctypedef struct  gdf_segmented_radixsort_plan_type:
        pass

    ctypedef enum order_by_type:
        GDF_ORDER_ASC,
        GDF_ORDER_DESC

    cdef gdf_error gdf_order_by(
        gdf_column** input_columns,
        int8_t* asc_desc,
        size_t num_inputs,
        gdf_column* output_indices,
        int flag_nulls_are_smallest
    ) except +

    cdef gdf_radixsort_plan_type* gdf_radixsort_plan(
        size_t num_items,
        int descending,
        unsigned begin_bit,
        unsigned end_bit
    ) except +

    cdef gdf_error gdf_radixsort_plan_setup(
        gdf_radixsort_plan_type *hdl,
        size_t sizeof_key,
        size_t sizeof_val
    ) except +

    cdef gdf_error gdf_radixsort_plan_free(
        gdf_radixsort_plan_type *hdl
    ) except +

    cdef gdf_error gdf_radixsort(
        gdf_radixsort_plan_type *hdl,
        gdf_column *keycol,
        gdf_column *valcol
    ) except +

    cdef gdf_segmented_radixsort_plan_type* gdf_segmented_radixsort_plan(
        size_t num_items,
        int descending,
        unsigned begin_bit,
        unsigned end_bit
    ) except +

    cdef gdf_error gdf_segmented_radixsort_plan_setup(
        gdf_segmented_radixsort_plan_type *hdl,
        size_t sizeof_key,
        size_t sizeof_val
    ) except +

    cdef gdf_error gdf_segmented_radixsort_plan_free(
        gdf_segmented_radixsort_plan_type *hdl
    ) except +

    cdef gdf_error gdf_segmented_radixsort(
        gdf_segmented_radixsort_plan_type *hdl,
        gdf_column *keycol,
        gdf_column *valcol,
        unsigned num_segments,
        unsigned *d_begin_offsets,
        unsigned *d_end_offsets
    ) except +
    
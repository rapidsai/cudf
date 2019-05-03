# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from libc.stdint cimport uintptr_t, int8_t
from libc.stdlib cimport malloc, free

import numpy as np
import pandas as pd
import pyarrow as pa
import itertools

from librmm_cffi import librmm as rmm


cpdef apply_order_by(in_cols, out_indices, ascending=True, na_position=1):
    '''
      Call gdf_order_by to retrieve a column of indices of the sorted order
      of rows.
    '''
    cdef gdf_column** input_columns = <gdf_column**>malloc(len(in_cols) * sizeof(gdf_column*))
    for idx, col in enumerate(in_cols):
        check_gdf_compatibility(col)
        input_columns[idx] = column_view_from_column(col)
    
    cdef uintptr_t asc_desc = get_ctype_ptr(ascending)

    cdef size_t num_inputs = len(in_cols)

    check_gdf_compatibility(out_indices)
    cdef gdf_column* output_indices = column_view_from_column(out_indices)

    cdef int flag_nulls_are_smallest = na_position

    cdef gdf_error result 
    
    with nogil:
        result = gdf_order_by(<gdf_column**> input_columns,
                              <int8_t*> asc_desc,
                              <size_t> num_inputs,
                              <gdf_column*> output_indices,
                              <int> flag_nulls_are_smallest)
    
    check_gdf_error(result)


cpdef digitize(column, bins, right=False):
    check_gdf_compatibility(column)
    cdef gdf_column* in_col = column_view_from_column(column)

    check_gdf_compatibility(bins)
    cdef gdf_column* bins_col = column_view_from_column(bins)

    cdef bool cright = right
    cdef gdf_error result
    out = rmm.device_array(len(column), dtype=np.int32)
    cdef uintptr_t out_ptr = get_ctype_ptr(out)

    with nogil:
        result = gdf_digitize(<gdf_column*> in_col,
                              <gdf_column*> bins_col,
                              <bool> cright,
                              <gdf_index_type*> out_ptr)

    check_gdf_error(result)
    return out


def apply_segsort(col_keys, col_vals, segments, descending=False,
                  plan=None):
    """Inplace segemented sort

    Parameters
    ----------
    col_keys : Column
    col_vals : Column
    segments : device array
    """
    # prepare
    nelem = len(col_keys)
    if nelem == segments.size:
        # As many seguments as there are elements.
        # Nothing to do.
        return

    if plan is None:
        plan = SegmentedRadixSortPlan(
            nelem,
            col_keys.dtype,
            col_vals.dtype,
            descending=descending
        )

    plan.sort(
        segments,
        col_keys,
        col_vals
    )
    return plan


class SegmentedRadixSortPlan(object):
    def __init__(self, nelem, key_dtype, val_dtype, descending=False):
        begin_bit = 0
        self.sizeof_key = key_dtype.itemsize
        self.sizeof_val = val_dtype.itemsize
        end_bit = self.sizeof_key * 8
        cdef uintptr_t plan
        cdef size_t c_nelem = nelem
        cdef int c_descending = descending
        cdef unsigned c_begin_bit = begin_bit
        cdef unsigned c_end_bit = end_bit
        with nogil:
            plan = <uintptr_t>gdf_segmented_radixsort_plan(
                c_nelem,
                c_descending,
                c_begin_bit,
                c_end_bit
            )
        self.plan = int(plan)
        self.nelem = nelem
        self.is_closed = False
        self.setup()

    def __del__(self):
        if not self.is_closed:
            self.close()

    def close(self):
        cdef uintptr_t c_plan = self.plan
        with nogil:
            gdf_segmented_radixsort_plan_free(
                <gdf_segmented_radixsort_plan_type*>c_plan
            )
        self.is_closed = True
        self.plan = None

    def setup(self):
        cdef uintptr_t c_plan = self.plan
        cdef size_t c_sizeof_key = self.sizeof_key
        cdef size_t c_sizeof_val = self.sizeof_val
        with nogil:
            gdf_segmented_radixsort_plan_setup(
                <gdf_segmented_radixsort_plan_type*>c_plan,
                c_sizeof_key,
                c_sizeof_val
            )
        self.plan = int(c_plan)

    def sort(self, segments, col_keys, col_vals):
        cdef gdf_column* c_col_keys = column_view_from_column(col_keys)
        cdef gdf_column* c_col_vals = column_view_from_column(col_vals)
    
        seg_dtype = np.uint32
        segsize_limit = 2 ** 16 - 1

        d_fullsegs = rmm.device_array(segments.size + 1, dtype=seg_dtype)
        d_begins = d_fullsegs[:-1]
        d_ends = d_fullsegs[1:]

        # Note: .astype is required below because .copy_to_device
        #       is just a plain memcpy
        d_begins.copy_to_device(cudautils.astype(segments, dtype=seg_dtype))
        d_ends[-1:].copy_to_device(np.require([self.nelem], dtype=seg_dtype))

        cdef uintptr_t c_plan = self.plan
        cdef uintptr_t d_begins_ptr
        cdef uintptr_t d_end_ptr
        cdef unsigned segsize

        # The following is to handle the segument size limit due to
        # max CUDA grid size.
        range0 = range(0, segments.size, segsize_limit)
        range1 = itertools.chain(range0[1:], [segments.size])
        for s, e in zip(range0, range1):
            d_begins_ptr = get_ctype_ptr(d_begins[s:])
            d_end_ptr = get_ctype_ptr(d_ends[s:])
            segsize = e - s
            with nogil:
                gdf_segmented_radixsort(
                    <gdf_segmented_radixsort_plan_type*>c_plan,
                    c_col_keys,
                    c_col_vals,
                    segsize,
                    <unsigned*>d_begins_ptr,
                    <unsigned*>d_end_ptr
                )

        self.plan = int(c_plan)
        free(c_col_keys)
        free(c_col_vals)

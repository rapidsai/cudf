# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from libc.stdlib cimport free
from libc.stdint cimport uintptr_t, uint32_t
from libcpp.vector cimport vector


def hash_columns(columns, result, initial_hash_values=None):
    """Hash the *columns* and store in *result*.
    Returns *result*
    """
    assert len(columns) > 0
    assert result.dtype == np.int32
    # No-op for 0-sized
    if len(result) == 0:
        return result
    cdef vector[gdf_column*] c_col_input
    for col in columns:
        c_col_input.push_back(column_view_from_column(col))
    cdef gdf_column* c_col_out = column_view_from_column(result)
    cdef int ncols = len(columns)
    cdef gdf_hash_func hashfn = GDF_HASH_MURMUR3
    cdef uintptr_t c_initial_hash_values
    if initial_hash_values is None:
        c_initial_hash_values = 0
    else:
        c_initial_hash_values = get_ctype_ptr(initial_hash_values)
    
    with nogil:
        err = gdf_hash(
            ncols,
            c_col_input.data(),
            hashfn,
            <uint32_t*>c_initial_hash_values,
            c_col_out
        )

    check_gdf_error(err)

    free(c_col_out)
    for c_col in c_col_input:
        free(c_col)

    return result


def hash_partition(input_columns, key_indices, nparts, output_columns):
    """Partition the input_columns by the hash values on the keys.

    Parameters
    ----------
    input_columns : sequence of Column
    key_indices : sequence of int
        Indices into `input_columns` that indicates the key columns.
    nparts : int
        number of partitions

    Returns
    -------
    partition_offsets : list of int
        Each index indicates the start of a partition.
    """
    assert len(input_columns) == len(output_columns)

    cdef int c_len_col_inputs = len(input_columns)
    cdef vector[gdf_column*] c_col_input
    for col in input_columns:
        c_col_input.push_back(column_view_from_column(col))
    cdef vector[int] c_key_indices = key_indices
    cdef int c_len_key_indices = len(key_indices)
    cdef int c_nparts = nparts
    cdef vector[gdf_column*] c_col_output
    for col in output_columns:
        c_col_output.push_back(column_view_from_column(col))
    cdef vector[int] offsets = vector[int](c_nparts)
    cdef gdf_hash_func hashfn = GDF_HASH_MURMUR3

    with nogil:
        err = gdf_hash_partition(
            c_len_col_inputs,
            c_col_input.data(),
            c_key_indices.data(),
            c_len_key_indices,
            c_nparts,
            c_col_output.data(),
            offsets.data(),
            hashfn
        )

    check_gdf_error(err)

    for c_col in c_col_input:
        free(c_col)
    for c_col in c_col_output:
        free(c_col)

    offsets = list(offsets)
    return offsets


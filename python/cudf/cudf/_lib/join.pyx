# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.includes.join cimport *
from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._lib.utils cimport *
from cudf._lib.utils import *

from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.stdlib cimport free
cimport cython


@cython.boundscheck(False)
cpdef join(col_lhs, col_rhs, left_on, right_on, how, method):
    """
      Call gdf join for full outer, inner and left joins.
      Returns a list of tuples [(column, valid, name), ...]
    """

    # TODO: `context` leaks if exiting this function prematurely
    cdef gdf_context* context = create_context_view(0, method, 0, 0, 0,
                                                    'null_as_largest', False)

    if how not in ['left', 'inner', 'outer']:
        msg = "new join api only supports left, inner or outer"
        raise ValueError(msg)

    left_idx = []
    right_idx = []

    assert(len(left_on) == len(right_on))

    cdef cudf_table *list_lhs = table_from_columns(col_lhs.values())
    cdef cudf_table *list_rhs = table_from_columns(col_rhs.values())
    cdef vector[int] left_on_ind
    cdef vector[int] right_on_ind
    cdef vector[pair[int, int]] columns_in_common

    result_col_names = []  # Preserve the order of the column names

    for name, col in col_lhs.items():
        check_gdf_compatibility(col)
        result_col_names.append(name)

    for name in left_on:
        # This will ensure that the column name is valid
        col_lhs[name]
        left_on_ind.push_back(list(col_lhs.keys()).index(name))
        if (name in right_on and
           (left_on.index(name) == right_on.index(name))):
            columns_in_common.push_back(pair[int, int](
                list(col_lhs.keys()).index(name),
                list(col_rhs.keys()).index(name)))

    for name in right_on:
        # This will ensure that the column name is valid
        col_rhs[name]
        right_on_ind.push_back(list(col_rhs.keys()).index(name))

    for name, col in col_rhs.items():
        check_gdf_compatibility(col)
        if not ((name in left_on) and (name in right_on)
           and (left_on.index(name) == right_on.index(name))):
            result_col_names.append(name)

    cdef cudf_table result

    with nogil:
        if how == 'left':
            result = left_join(
                list_lhs[0],
                list_rhs[0],
                left_on_ind,
                right_on_ind,
                columns_in_common,
                <cudf_table*> NULL,
                context
            )

        elif how == 'inner':
            result = inner_join(
                list_lhs[0],
                list_rhs[0],
                left_on_ind,
                right_on_ind,
                columns_in_common,
                <cudf_table*> NULL,
                context
            )

        elif how == 'outer':
            result = full_join(
                list_lhs[0],
                list_rhs[0],
                left_on_ind,
                right_on_ind,
                columns_in_common,
                <cudf_table*> NULL,
                context
            )

    res = columns_from_table(&result)

    free(context)

    del list_lhs
    del list_rhs

    return list(zip(res, result_col_names))

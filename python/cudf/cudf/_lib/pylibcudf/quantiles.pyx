# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.quantiles cimport (
    quantile as cpp_quantile,
    quantiles as cpp_quantiles,
)
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport null_order, order, sorted

from .column cimport Column
from .table cimport Table
from .types cimport interpolation


cpdef Column quantile(
    Column input,
    const double[:] q,
    interpolation interp = interpolation.LINEAR,
    Column ordered_indices = None,
    bool exact=True
):
    """Computes quantiles with interpolation.

    Computes the specified quantiles by interpolating values between which they lie,
    using the interpolation strategy specified in interp.

    Parameters
    ----------
    q: array-like that implements buffer-protocol
        The quantiles to calculate in range [0,1]
    interp: Interpolation, default Interpolation.LINEAR
        The strategy used to select between values adjacent to a specified quantile.
    ordered_indices: Column, default empty column
        The column containing the sorted order of input.

        If empty, all input values are used in existing order.
        Indices must be in range [0, input.size()), but are not required to be unique.
        Values not indexed by this column will be ignored.
    exact: bool, default True
        Returns doubles if True. Otherwise, returns same type as input

    Returns
    -------
    Column
        A Column containing specified quantiles, with nulls for indeterminable values
    """
    cdef:
        unique_ptr[column] c_result
        vector[double] q_vec
        column_view ordered_indices_view

    if ordered_indices is None:
        ordered_indices_view = column_view()
    else:
        ordered_indices_view = ordered_indices.view()

    # Copy from memoryview into vector
    if len(q) > 0:
        q_vec.assign(&q[0], &q[0] + len(q))

    with nogil:
        c_result = move(
            cpp_quantile(
                input.view(),
                q_vec,
                interp,
                ordered_indices_view,
                exact,
            )
        )

    return Column.from_libcudf(move(c_result))


cpdef Table quantiles(
    Table input,
    const double[:] q,
    interpolation interp = interpolation.NEAREST,
    sorted is_input_sorted = sorted.NO,
    # cython-lint complains that this a dangerous default value but
    # we don't modify these parameters, and so should be good to go
    list column_order = [],  # no-cython-lint
    list null_precedence = [],  # no-cython-lint
):
    """Computes row quantiles with interpolation.

    Computes the specified quantiles by retrieving the row corresponding to the
    specified quantiles. In the event a quantile lies in between rows, the specified
    interpolation strategy is used to pick between the rows.

    Parameters
    ----------
    q: array-like that implements buffer-protocol
        The quantiles to calculate in range [0,1]
    interp: Interpolation, default Interpolation.LINEAR
        The strategy used to select between values adjacent to a specified quantile.

        Must be a non-arithmetic interpolation strategy
        (i.e. one of
        {`Interpolation.HIGHER`, `Interpolation.LOWER`, `Interpolation.NEAREST`})
    is_input_sorted: Sorted, default Sorted.NO
        Whether the input table has been pre-sorted or not.
    column_order: list, default []
        A list of `Order` enums, indicating the desired sort order for each column.

        Ignored if `is_input_sorted` is `Sorted.YES`
    null_precedence: list, default []
        A list of `NullOrder` enums, indicating how nulls should be sorted.

        Ignored if `is_input_sorted` is `Sorted.YES`

    Returns
    -------
    Column
        A Column containing specified quantiles, with nulls for indeterminable values
    """
    cdef:
        unique_ptr[table] c_result
        vector[double] q_vec
        vector[order] column_order_vec = column_order
        vector[null_order] null_precedence_vec = null_precedence

    # Copy from memoryview into vector
    q_vec.assign(&q[0], &q[0] + len(q))

    with nogil:
        c_result = move(
            cpp_quantiles(
                input.view(),
                q_vec,
                interp,
                is_input_sorted,
                column_order_vec,
                null_precedence_vec,
            )
        )

    return Table.from_libcudf(move(c_result))

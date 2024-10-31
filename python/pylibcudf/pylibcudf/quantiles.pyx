# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.quantiles cimport (
    quantile as cpp_quantile,
    quantiles as cpp_quantiles,
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport null_order, order, sorted

from .column cimport Column
from .table cimport Table
from .types cimport interpolation


cpdef Column quantile(
    Column input,
    vector[double] q,
    interpolation interp = interpolation.LINEAR,
    Column ordered_indices = None,
    bool exact=True
):
    """Computes quantiles with interpolation.

    Computes the specified quantiles by interpolating values between which they lie,
    using the interpolation strategy specified in interp.

    For details see :cpp:func:`quantile`.

    Parameters
    ----------
    input: Column
        The Column to calculate quantiles on.
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

    For details, see :cpp:func:`quantile`.

    Returns
    -------
    Column
        A Column containing specified quantiles, with nulls for indeterminable values
    """
    cdef:
        unique_ptr[column] c_result
        column_view ordered_indices_view

    if ordered_indices is None:
        ordered_indices_view = column_view()
    else:
        ordered_indices_view = ordered_indices.view()

    with nogil:
        c_result = cpp_quantile(
            input.view(),
            q,
            interp,
            ordered_indices_view,
            exact,
        )

    return Column.from_libcudf(move(c_result))


cpdef Table quantiles(
    Table input,
    vector[double] q,
    interpolation interp = interpolation.NEAREST,
    sorted is_input_sorted = sorted.NO,
    list column_order = None,
    list null_precedence = None,
):
    """Computes row quantiles with interpolation.

    Computes the specified quantiles by retrieving the row corresponding to the
    specified quantiles. In the event a quantile lies in between rows, the specified
    interpolation strategy is used to pick between the rows.

    For details see :cpp:func:`quantiles`.

    Parameters
    ----------
    input: Table
        The Table to calculate row quantiles on.
    q: array-like
        The quantiles to calculate in range [0,1]
    interp: Interpolation, default Interpolation.NEAREST
        The strategy used to select between values adjacent to a specified quantile.

        Must be a non-arithmetic interpolation strategy
        (i.e. one of
        {`Interpolation.HIGHER`, `Interpolation.LOWER`, `Interpolation.NEAREST`})
    is_input_sorted: Sorted, default Sorted.NO
        Whether the input table has been pre-sorted or not.
    column_order: list, default None
        A list of `Order` enums,
        indicating the desired sort order for each column.
        By default, will sort all columns so that they are in ascending order.

        Ignored if `is_input_sorted` is `Sorted.YES`
    null_precedence: list, default None
        A list of `NullOrder` enums,
        indicating how nulls should be sorted.
        By default, will sort all columns so that nulls appear before
        all other elements.

        Ignored if `is_input_sorted` is `Sorted.YES`

    For details, see :cpp:func:`quantiles`.

    Returns
    -------
    Column
        A Column containing specified quantiles, with nulls for indeterminable values
    """
    cdef:
        unique_ptr[table] c_result
        vector[order] column_order_vec
        vector[null_order] null_precedence_vec

    if column_order is not None:
        column_order_vec = column_order
    if null_precedence is not None:
        null_precedence_vec = null_precedence

    with nogil:
        c_result = cpp_quantiles(
            input.view(),
            q,
            interp,
            is_input_sorted,
            column_order_vec,
            null_precedence_vec,
        )

    return Table.from_libcudf(move(c_result))

# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.lists cimport (
    contains as cpp_contains,
    explode as cpp_explode,
    filling as cpp_filling,
    gather as cpp_gather,
    reverse as cpp_reverse,
    set_operations as cpp_set_operations,
)
from cudf._lib.pylibcudf.libcudf.lists.combine cimport (
    concatenate_list_elements as cpp_concatenate_list_elements,
    concatenate_null_policy,
    concatenate_rows as cpp_concatenate_rows,
)
from cudf._lib.pylibcudf.libcudf.lists.count_elements cimport (
    count_elements as cpp_count_elements,
)
from cudf._lib.pylibcudf.libcudf.lists.extract cimport (
    extract_list_element as cpp_extract_list_element,
)
from cudf._lib.pylibcudf.libcudf.lists.sorting cimport (
    sort_lists as cpp_sort_lists,
    stable_sort_lists as cpp_stable_sort_lists,
)
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport (
    nan_equality,
    null_equality,
    null_order,
    order,
    size_type,
)
from cudf._lib.pylibcudf.lists cimport ColumnOrScalar, ColumnOrSizeType

from .column cimport Column, ListColumnView
from .scalar cimport Scalar
from .table cimport Table


cpdef Table explode_outer(Table input, size_type explode_column_idx):
    """Explode a column of lists into rows.

    All other columns will be duplicated for each element in the list.

    Parameters
    ----------
    input : Table
        The input table
    explode_column_idx : int
        The index of the column to explode

    Returns
    -------
    Table
        A new table with the exploded column
    """
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_explode.explode_outer(input.view(), explode_column_idx))

    return Table.from_libcudf(move(c_result))


cpdef Column concatenate_rows(Table input):
    """Concatenate multiple lists columns into a single lists column row-wise.

    Parameters
    ----------
    input : Table
        The input table

    Returns
    -------
    Table
        A new Column of concatenated rows
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_concatenate_rows(input.view()))

    return Column.from_libcudf(move(c_result))


cpdef Column concatenate_list_elements(Column input, bool dropna):
    """Concatenate multiple lists on the same row into a single list.

    Parameters
    ----------
    input : Column
        The input column
    dropna : bool
        If true, null list elements will be ignored
        from concatenation. Otherwise any input null values will result in
        the corresponding output row being set to null.

    Returns
    -------
    Column
        A new Column of concatenated list elements
    """
    cdef concatenate_null_policy null_policy = (
        concatenate_null_policy.IGNORE if dropna
        else concatenate_null_policy.NULLIFY_OUTPUT_ROW
    )
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_concatenate_list_elements(
            input.view(),
            null_policy,
        ))

    return Column.from_libcudf(move(c_result))


cpdef Column contains(Column input, ColumnOrScalar search_key):
    """Create a column of bool values indicating whether
    the search_key is contained in the input.

    ``search_key`` may be a
    :py:class:`~cudf._lib.pylibcudf.column.Column` or a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`.

    For details, see :cpp:func:`contains`.

    Parameters
    ----------
    input : Column
        The input column.
    search_key : Union[Column, Scalar]
        The search key.

    Returns
    -------
    Column
        A new Column of bools indicating if the search_key was
        found in the list column.
    """
    cdef unique_ptr[column] c_result
    cdef ListColumnView list_view = input.list_view()

    if not isinstance(search_key, (Column, Scalar)):
        raise TypeError("Must pass a Column or Scalar")

    with nogil:
        c_result = move(cpp_contains.contains(
            list_view.view(),
            search_key.view() if ColumnOrScalar is Column else dereference(
                search_key.get()
            ),
        ))
    return Column.from_libcudf(move(c_result))


cpdef Column contains_nulls(Column input):
    """Create a column of bool values indicating whether
    each row in the lists column contains a null value.

    Parameters
    ----------
    input : Column
        The input column.

    Returns
    -------
    Column
        A new Column of bools indicating if the list column
        contains a null value.
    """
    cdef unique_ptr[column] c_result
    cdef ListColumnView list_view = input.list_view()
    with nogil:
        c_result = move(cpp_contains.contains_nulls(list_view.view()))
    return Column.from_libcudf(move(c_result))


cpdef Column index_of(Column input, ColumnOrScalar search_key, bool find_first_option):
    """Create a column of index values indicating the position of a search
    key row within the corresponding list row in the lists column.

    ``search_key`` may be a
    :py:class:`~cudf._lib.pylibcudf.column.Column` or a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`.

    For details, see :cpp:func:`index_of`.

    Parameters
    ----------
    input : Column
        The input column.
    search_key : Union[Column, Scalar]
        The search key.
    find_first_option : bool
        If true, index_of returns the first match.
        Otherwise the last match is returned.

    Returns
    -------
    Column
        A new Column of index values that indicate where in the
        list column tthe search_key was found. An index value
        of -1 indicates that the search_key was not found.
    """
    cdef unique_ptr[column] c_result
    cdef ListColumnView list_view = input.list_view()
    cdef cpp_contains.duplicate_find_option find_option = (
        cpp_contains.duplicate_find_option.FIND_FIRST if find_first_option
        else cpp_contains.duplicate_find_option.FIND_LAST
    )

    with nogil:
        c_result = move(cpp_contains.index_of(
            list_view.view(),
            search_key.view() if ColumnOrScalar is Column else dereference(
                search_key.get()
            ),
            find_option,
        ))
    return Column.from_libcudf(move(c_result))


cpdef Column reverse(Column input):
    """Reverse the element order within each list of the input column.

    For details, see :cpp:func:`reverse`.

    Parameters
    ----------
    input : Column
        The input column.

    Returns
    -------
    Column
        A new Column with reversed lists.
    """
    cdef unique_ptr[column] c_result
    cdef ListColumnView list_view = input.list_view()

    with nogil:
        c_result = move(cpp_reverse.reverse(
            list_view.view(),
        ))
    return Column.from_libcudf(move(c_result))


cpdef Column segmented_gather(Column input, Column gather_map_list):
    """Create a column with elements gathered based on the indices in gather_map_list

    For details, see :cpp:func:`segmented_gather`.

    Parameters
    ----------
    input : Column
        The input column.
    gather_map_list : Column
        The indices of the lists column to gather.

    Returns
    -------
    Column
        A new Column with elements in list of rows
        gathered based on gather_map_list
    """

    cdef unique_ptr[column] c_result
    cdef ListColumnView list_view1 = input.list_view()
    cdef ListColumnView list_view2 = gather_map_list.list_view()

    with nogil:
        c_result = move(cpp_gather.segmented_gather(
            list_view1.view(),
            list_view2.view(),
        ))
    return Column.from_libcudf(move(c_result))


cpdef Column extract_list_element(Column input, ColumnOrSizeType index):
    """Create a column of extracted list elements.

    Parameters
    ----------
    input : Column
        The input column.
    index : Union[Column, size_type]
        The selection index or indices.

    Returns
    -------
    Column
        A new Column with elements extracted.
    """
    cdef unique_ptr[column] c_result
    cdef ListColumnView list_view = input.list_view()

    with nogil:
        c_result = move(cpp_extract_list_element(
            list_view.view(),
            index.view() if ColumnOrSizeType is Column else index,
        ))
    return Column.from_libcudf(move(c_result))


cpdef Column count_elements(Column input):
    """Count the number of rows in each
    list element in the given lists column.
    For details, see :cpp:func:`count_elements`.

    Parameters
    ----------
    input : Column
        The input column

    Returns
    -------
    Column
        A new Column of the lengths of each list element
    """
    cdef ListColumnView list_view = input.list_view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_count_elements(list_view.view()))

    return Column.from_libcudf(move(c_result))


cpdef Column sequences(Column starts, Column sizes, Column steps = None):
    """Create a lists column in which each row contains a sequence of
    values specified by a tuple of (start, step, size) parameters.

    For details, see :cpp:func:`sequences`.

    Parameters
    ----------
    starts : Column
        First values in the result sequences.
    sizes : Column
        Numbers of values in the result sequences.
    steps : Optional[Column]
        Increment values for the result sequences.

    Returns
    -------
    Column
        The result column containing generated sequences.
    """
    cdef unique_ptr[column] c_result

    if steps is not None:
        with nogil:
            c_result = move(cpp_filling.sequences(
                starts.view(),
                steps.view(),
                sizes.view(),
            ))
    else:
        with nogil:
            c_result = move(cpp_filling.sequences(
                starts.view(),
                sizes.view(),
            ))
    return Column.from_libcudf(move(c_result))

cpdef Column sort_lists(
    Column input,
    bool ascending,
    null_order na_position,
    bool stable = False
):
    """Sort the elements within a list in each row of a list column.

    For details, see :cpp:func:`sort_lists`.

    Parameters
    ----------
    input : Column
        The input column.
    ascending : bool
        If true, the sort order is ascending. Otherwise, the sort order is descending.
    na_position : NullOrder
        If na_position equals NullOrder.FIRST, then the null values in the output
        column are placed first. Otherwise, they are be placed after.
    stable: bool
        If true :cpp:func:`stable_sort_lists` is used, Otherwise,
        :cpp:func:`sort_lists` is used.

    Returns
    -------
    Column
        A new Column with elements in each list sorted.
    """
    cdef unique_ptr[column] c_result
    cdef ListColumnView list_view = input.list_view()

    cdef order c_sort_order = (
        order.ASCENDING if ascending else order.DESCENDING
    )

    with nogil:
        if stable:
            c_result = move(cpp_stable_sort_lists(
                    list_view.view(),
                    c_sort_order,
                    na_position,
            ))
        else:
            c_result = move(cpp_sort_lists(
                    list_view.view(),
                    c_sort_order,
                    na_position,
            ))
    return Column.from_libcudf(move(c_result))


cpdef Column difference_distinct(
    Column lhs,
    Column rhs,
    bool nulls_equal=True,
    bool nans_equal=True
):
    """Create a column of index values indicating the position of a search
    key row within the corresponding list row in the lists column.

    For details, see :cpp:func:`difference_distinct`.

    Parameters
    ----------
    lhs : Column
        The input lists column of elements that may be included.
    rhs : Column
        The input lists column of elements to exclude.
    nulls_equal : bool, default True
        If true, null elements are considered equal. Otherwise, unequal.
    nans_equal : bool, default True
        If true, libcudf will treat nan elements from {-nan, +nan}
        as equal. Otherwise, unequal. Otherwise, unequal.

    Returns
    -------
    Column
        A lists column containing the difference results.
    """
    cdef unique_ptr[column] c_result
    cdef ListColumnView lhs_view = lhs.list_view()
    cdef ListColumnView rhs_view = rhs.list_view()

    cdef null_equality c_nulls_equal = (
        null_equality.EQUAL if nulls_equal else null_equality.UNEQUAL
    )
    cdef nan_equality c_nans_equal = (
        nan_equality.ALL_EQUAL if nans_equal else nan_equality.UNEQUAL
    )

    with nogil:
        c_result = move(cpp_set_operations.difference_distinct(
            lhs_view.view(),
            rhs_view.view(),
            c_nulls_equal,
            c_nans_equal,
        ))
    return Column.from_libcudf(move(c_result))


cpdef Column have_overlap(
    Column lhs,
    Column rhs,
    bool nulls_equal=True,
    bool nans_equal=True
):
    """Check if lists at each row of the given lists columns overlap.

    For details, see :cpp:func:`have_overlap`.

    Parameters
    ----------
    lhs : Column
        The input lists column for one side.
    rhs : Column
        The input lists column for the other side.
    nulls_equal : bool, default True
        If true, null elements are considered equal. Otherwise, unequal.
    nans_equal : bool, default True
        If true, libcudf will treat nan elements from {-nan, +nan}
        as equal. Otherwise, unequal. Otherwise, unequal.

    Returns
    -------
    Column
        A column containing the check results.
    """
    cdef unique_ptr[column] c_result
    cdef ListColumnView lhs_view = lhs.list_view()
    cdef ListColumnView rhs_view = rhs.list_view()

    cdef null_equality c_nulls_equal = (
        null_equality.EQUAL if nulls_equal else null_equality.UNEQUAL
    )
    cdef nan_equality c_nans_equal = (
        nan_equality.ALL_EQUAL if nans_equal else nan_equality.UNEQUAL
    )

    with nogil:
        c_result = move(cpp_set_operations.have_overlap(
            lhs_view.view(),
            rhs_view.view(),
            c_nulls_equal,
            c_nans_equal,
        ))
    return Column.from_libcudf(move(c_result))


cpdef Column intersect_distinct(
    Column lhs,
    Column rhs,
    bool nulls_equal=True,
    bool nans_equal=True
):
    """Create a lists column of distinct elements common to two input lists columns.

    For details, see :cpp:func:`intersect_distinct`.

    Parameters
    ----------
    lhs : Column
        The input lists column of elements that may be included.
    rhs : Column
        The input lists column of elements to exclude.
    nulls_equal : bool, default True
        If true, null elements are considered equal. Otherwise, unequal.
    nans_equal : bool, default True
        If true, libcudf will treat nan elements from {-nan, +nan}
        as equal. Otherwise, unequal. Otherwise, unequal.

    Returns
    -------
    Column
        A lists column containing the intersection results.
    """
    cdef unique_ptr[column] c_result
    cdef ListColumnView lhs_view = lhs.list_view()
    cdef ListColumnView rhs_view = rhs.list_view()

    cdef null_equality c_nulls_equal = (
        null_equality.EQUAL if nulls_equal else null_equality.UNEQUAL
    )
    cdef nan_equality c_nans_equal = (
        nan_equality.ALL_EQUAL if nans_equal else nan_equality.UNEQUAL
    )

    with nogil:
        c_result = move(cpp_set_operations.intersect_distinct(
            lhs_view.view(),
            rhs_view.view(),
            c_nulls_equal,
            c_nans_equal,
        ))
    return Column.from_libcudf(move(c_result))


cpdef Column union_distinct(
    Column lhs,
    Column rhs,
    bool nulls_equal=True,
    bool nans_equal=True
):
    """Create a lists column of distinct elements found in
    either of two input lists columns.

    For details, see :cpp:func:`union_distinct`.

    Parameters
    ----------
    lhs : Column
        The input lists column of elements that may be included.
    rhs : Column
        The input lists column of elements to exclude.
    nulls_equal : bool, default True
        If true, null elements are considered equal. Otherwise, unequal.
    nans_equal : bool, default True
        If true, libcudf will treat nan elements from {-nan, +nan}
        as equal. Otherwise, unequal. Otherwise, unequal.

    Returns
    -------
    Column
        A lists column containing the union results.
    """
    cdef unique_ptr[column] c_result
    cdef ListColumnView lhs_view = lhs.list_view()
    cdef ListColumnView rhs_view = rhs.list_view()

    cdef null_equality c_nulls_equal = (
        null_equality.EQUAL if nulls_equal else null_equality.UNEQUAL
    )
    cdef nan_equality c_nans_equal = (
        nan_equality.ALL_EQUAL if nans_equal else nan_equality.UNEQUAL
    )

    with nogil:
        c_result = move(cpp_set_operations.union_distinct(
            lhs_view.view(),
            rhs_view.view(),
            c_nulls_equal,
            c_nans_equal,
        ))
    return Column.from_libcudf(move(c_result))

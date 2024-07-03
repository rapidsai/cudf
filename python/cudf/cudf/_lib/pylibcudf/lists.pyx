# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.lists cimport (
    contains as cpp_contains,
    explode as cpp_explode,
    set_operations as cpp_set_operations,
)
from cudf._lib.pylibcudf.libcudf.lists.combine cimport (
    concatenate_list_elements as cpp_concatenate_list_elements,
    concatenate_null_policy,
    concatenate_rows as cpp_concatenate_rows,
)
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport (
    nan_equality,
    null_equality,
    size_type,
)
from cudf._lib.pylibcudf.lists cimport ColumnOrScalar

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
    nulls_equal : bool
        If true, null elements are considered equal. Otherwise, unequal.
    nans_equal : bool
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
    nulls_equal : bool
        If true, null elements are considered equal. Otherwise, unequal.
    nans_equal : bool
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
    nulls_equal : bool
        If true, null elements are considered equal. Otherwise, unequal.
    nans_equal : bool
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
    nulls_equal : bool
        If true, null elements are considered equal. Otherwise, unequal.
    nans_equal : bool
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

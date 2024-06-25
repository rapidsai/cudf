# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.lists cimport (
    contains as cpp_contains,
    explode as cpp_explode,
)
from cudf._lib.pylibcudf.libcudf.lists.combine cimport (
    concatenate_list_elements as cpp_concatenate_list_elements,
    concatenate_null_policy,
    concatenate_rows as cpp_concatenate_rows,
)
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport size_type

from .column cimport Column, ListColumnView
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
    """Create a column of bool values based upon the search_key,
    indicating whether the search_key is contained in the input.

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
    """Create a column of values indicating the position of a search
    key row within the corresponding list row in the lists column.

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
        A new Column of index values that if the search_key
        was found in the list column. An index value of -1
        indicates that the search_key weas not found.
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

# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
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
from cudf._lib.pylibcudf.libcudf.lists.lists_column_view cimport (
    lists_column_view,
)
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.scalar cimport DeviceScalar

from .column cimport Column
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
    """Create a column of bool values based upon the search key.

    Parameters
    ----------
    input : Column
        The input column.
    search_key : Union[Column, Scalar]
        The search key.

    Returns
    -------
    Column
        A new Column of bools
    """
    cdef unique_ptr[column] c_result
    cdef shared_ptr[lists_column_view] list_view = (
        make_shared[lists_column_view](input.view())
    )
    cdef const scalar* search_key_value = NULL

    if ColumnOrScalar is Column:
        with nogil:
            c_result = move(cpp_contains.contains(
                list_view.get()[0],
                search_key.view(),
            ))
    elif ColumnOrScalar is DeviceScalar:
        search_key_value = search_key.get_raw_ptr()
        with nogil:
            c_result = move(cpp_contains.contains(
                list_view.get()[0],
                search_key_value[0],
            ))
    else:
        search_key_value = search_key.get()
        with nogil:
            c_result = move(cpp_contains.contains(
                list_view.get()[0],
                search_key_value[0],
            ))
    return Column.from_libcudf(move(c_result))

# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.lists cimport explode as cpp_explode
from cudf._lib.pylibcudf.libcudf.lists.extract cimport extract_list_element as cpp_extract_list_element
from cudf._lib.pylibcudf.libcudf.lists.combine cimport (
    concatenate_list_elements as cpp_concatenate_list_elements,
    concatenate_null_policy,
    concatenate_rows as cpp_concatenate_rows,
)
from cudf._lib.pylibcudf.libcudf.lists.lists_column_view cimport (
    lists_column_view,
)
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport size_type

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

    Returns
    -------
    Column
        A new Column of concatenated list elements
    dropna : bool
        If true, null list elements will be ignored
        from concatenation. Otherwise any input null values will result in
        the corresponding output row being set to null.
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


cpdef Column extract_list_element(Column input, ColumnOrSizeType index):
    """Create a column of extracted list elements.
    Parameters
    ----------
    input : Column
        The input column.
    index : Union[Column, size_type]
        The selection index or indicies.
    Returns
    -------
    Column
        A new Column of bools
    """
    cdef unique_ptr[column] c_result
    cdef shared_ptr[lists_column_view] list_view = (
        make_shared[lists_column_view](input.view())
    )
    if ColumnOrScalar is Column:
        with nogil:
            c_result = move(extract_list_element(list_view.get()[0], index.view()))
    else:
        with nogil:
            c_result = move(extract_list_element(list_view.get()[0], index))
    return Column.from_libcudf(move(c_result))

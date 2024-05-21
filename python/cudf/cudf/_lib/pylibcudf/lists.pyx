# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf.lists cimport explode as cpp_explode
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport size_type

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

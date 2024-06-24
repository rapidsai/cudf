# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.datetime cimport (
    extract_year as cpp_extract_year,
)

from .column cimport Column


cpdef Column extract_year(
    Column values
):
    """
    Extract the year from a datetime column.

    Parameters
    ----------
    values : Column
        The column to extract the year from.

    Returns
    -------
    Column
        Column with the extracted years.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_extract_year(values.view()))
    return Column.from_libcudf(move(result))

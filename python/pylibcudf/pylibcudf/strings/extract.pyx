# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport extract as cpp_extract
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.table cimport Table
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

__all__ = ["extract", "extract_all_record", "extract_single"]

cpdef Table extract(
    Column input, RegexProgram prog, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a table of strings columns where each column
    corresponds to the matching group specified in the given
    egex_program object.

    For details, see :cpp:func:`cudf::strings::extract`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    prog : RegexProgram
        Regex program instance
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Table
        Columns of strings extracted from the input column.
    """
    cdef unique_ptr[table] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_extract.extract(
            input.view(),
            prog.c_obj.get()[0],
            stream.view(),
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), stream, mr)


cpdef Column extract_all_record(
    Column input, RegexProgram prog, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a lists column of strings where each string column
    row corresponds to the matching group specified in the given
    regex_program object.

    For details, see :cpp:func:`cudf::strings::extract_all_record`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    prog : RegexProgram
        Regex program instance
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Lists column containing strings extracted from the input column
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_extract.extract_all_record(
            input.view(),
            prog.c_obj.get()[0],
            stream.view(),
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Column extract_single(
    Column input,
    RegexProgram prog,
    size_type group,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a column of strings where each string corresponds to the
    matching group specified in the given regex_program object.

    For details, see :cpp:func:`cudf::strings::extract_single`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    prog : RegexProgram
        Regex program instance
    group : size_type
        Index of the group number to extract
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column of strings extracted from the input column
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_extract.extract_single(
            input.view(),
            prog.c_obj.get()[0],
            group,
            stream.view(),
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)

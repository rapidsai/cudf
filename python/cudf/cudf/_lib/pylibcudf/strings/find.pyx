# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

cimport cudf._lib.cpp.strings.find as cpp_find
from cudf._lib.cpp.column.column cimport column
from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.scalar cimport Scalar

from cython.operator import dereference

from cudf._lib.cpp.scalar.scalar cimport string_scalar


cpdef Column find(
    Column input,
    ColumnOrScalar target,
    size_type start=0,
    size_type stop=-1
):
    cdef unique_ptr[column] result
    if ColumnOrScalar is Column:
        with nogil:
            result = move(
                cpp_find.find(
                    input.view(),
                    target.view(),
                    start
                )
            )
    elif ColumnOrScalar is Scalar:
        with nogil:
            result = move(
                cpp_find.find(
                    input.view(),
                    dereference(<string_scalar*>(target.c_obj.get())),
                    start,
                    stop
                )
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result))


cpdef Column rfind(
    Column input,
    Scalar target,
    size_type start=0,
    size_type stop=-1
):
    cdef unique_ptr[column] result
    with nogil:
        result = move(
            cpp_find.rfind(
                input.view(),
                dereference(<string_scalar*>(target.c_obj.get())),
                start,
                stop
            )
        )
    return Column.from_libcudf(move(result))


cpdef Column contains(
    Column input,
    ColumnOrScalar target,
):
    cdef unique_ptr[column] result
    if ColumnOrScalar is Column:
        with nogil:
            result = move(
                cpp_find.contains(
                    input.view(),
                    target.view()
                )
            )
    elif ColumnOrScalar is Scalar:
        with nogil:
            result = move(
                cpp_find.contains(
                    input.view(),
                    dereference(<string_scalar*>(target.c_obj.get()))
                )
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result))


cpdef Column starts_with(
    Column input,
    ColumnOrScalar target,
):
    cdef unique_ptr[column] result

    if ColumnOrScalar is Column:
        with nogil:
            result = move(
                cpp_find.starts_with(
                    input.view(),
                    target.view()
                )
            )
    elif ColumnOrScalar is Scalar:
        with nogil:
            result = move(
                cpp_find.starts_with(
                    input.view(),
                    dereference(<string_scalar*>(target.c_obj.get()))
                )
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result))

cpdef Column ends_with(
    Column input,
    ColumnOrScalar target,
):
    cdef unique_ptr[column] result
    if ColumnOrScalar is Column:
        with nogil:
            result = move(
                cpp_find.ends_with(
                    input.view(),
                    target.view()
                )
            )
    elif ColumnOrScalar is Scalar:
        with nogil:
            result = move(
                cpp_find.ends_with(
                    input.view(),
                    dereference(<string_scalar*>(target.c_obj.get()))
                )
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result))

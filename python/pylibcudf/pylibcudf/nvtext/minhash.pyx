# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.nvtext.minhash cimport (
    minhash as cpp_minhash,
    minhash64 as cpp_minhash64,
    word_minhash as cpp_word_minhash,
    word_minhash64 as cpp_word_minhash64,
)
from pylibcudf.libcudf.scalar.scalar cimport numeric_scalar
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar

from cython.operator import dereference


cpdef Column minhash(Column input, ColumnOrScalar seeds, size_type width=4):
    """
    Returns the minhash values for each string per seed.
    This function uses MurmurHash3_x86_32 for the hash algorithm.

    For details, see :cpp:func:`minhash`.

    Parameters
    ----------
    input : Column
        Strings column to compute minhash
    seeds : Column or Scalar
        Seed value(s) used for the hash algorithm.
    width : size_type
        Character width used for apply substrings;
        Default is 4 characters.

    Returns
    -------
    Column
        List column of minhash values for each string per seed
    """
    cdef unique_ptr[column] c_result

    if not isinstance(seeds, (Column, Scalar)):
        raise TypeError("Must pass a Column or Scalar")

    with nogil:
        c_result = cpp_minhash(
            input.view(),
            seeds.view() if ColumnOrScalar is Column else
            dereference(<numeric_scalar[uint32_t]*>seeds.c_obj.get()),
            width
        )

    return Column.from_libcudf(move(c_result))

cpdef Column minhash64(Column input, ColumnOrScalar seeds, size_type width=4):
    """
    Returns the minhash values for each string per seed.
    This function uses MurmurHash3_x64_128 for the hash algorithm.

    For details, see :cpp:func:`minhash64`.

    Parameters
    ----------
    input : Column
        Strings column to compute minhash
    seeds : Column or Scalar
        Seed value(s) used for the hash algorithm.
    width : size_type
        Character width used for apply substrings;
        Default is 4 characters.

    Returns
    -------
    Column
        List column of minhash values for each string per seed
    """
    cdef unique_ptr[column] c_result

    if not isinstance(seeds, (Column, Scalar)):
        raise TypeError("Must pass a Column or Scalar")

    with nogil:
        c_result = cpp_minhash64(
            input.view(),
            seeds.view() if ColumnOrScalar is Column else
            dereference(<numeric_scalar[uint64_t]*>seeds.c_obj.get()),
            width
        )

    return Column.from_libcudf(move(c_result))

cpdef Column word_minhash(Column input, Column seeds):
    """
    Returns the minhash values for each row of strings per seed.
    This function uses MurmurHash3_x86_32 for the hash algorithm.

    For details, see :cpp:func:`word_minhash`.

    Parameters
    ----------
    input : Column
        Lists column of strings to compute minhash
    seeds : Column or Scalar
        Seed values used for the hash algorithm.

    Returns
    -------
    Column
        List column of minhash values for each string per seed
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_word_minhash(
            input.view(),
            seeds.view()
        )

    return Column.from_libcudf(move(c_result))

cpdef Column word_minhash64(Column input, Column seeds):
    """
    Returns the minhash values for each row of strings per seed.
    This function uses MurmurHash3_x64_128 for the hash algorithm though
    only the first 64-bits of the hash are used in computing the output.

    For details, see :cpp:func:`word_minhash64`.

    Parameters
    ----------
    input : Column
        Lists column of strings to compute minhash
    seeds : Column or Scalar
        Seed values used for the hash algorithm.

    Returns
    -------
    Column
        List column of minhash values for each string per seed
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_word_minhash64(
            input.view(),
            seeds.view()
        )

    return Column.from_libcudf(move(c_result))

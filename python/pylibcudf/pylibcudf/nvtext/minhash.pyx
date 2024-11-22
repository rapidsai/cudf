# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.nvtext.minhash cimport (
    minhash as cpp_minhash,
    minhash64 as cpp_minhash64,
)
from pylibcudf.libcudf.types cimport size_type

__all__ = [
    "minhash",
    "minhash64",
]

cpdef Column minhash(
    Column input,
    uint32_t seed,
    Column a,
    Column b,
    size_type width
):
    """
    Returns the minhash values for each string.
    This function uses MurmurHash3_x86_32 for the hash algorithm.

    For details, see :cpp:func:`minhash`.

    Parameters
    ----------
    input : Column
        Strings column to compute minhash
    seed : uint32_t
        Seed used for the hash function
    a : Column
        1st parameter value used for the minhash algorithm.
    b : Column
        2nd parameter value used for the minhash algorithm.
    width : size_type
        Character width used for apply substrings;

    Returns
    -------
    Column
        List column of minhash values for each string per seed
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_minhash(
            input.view(),
            seed,
            a.view(),
            b.view(),
            width
        )

    return Column.from_libcudf(move(c_result))

cpdef Column minhash64(
    Column input,
    uint64_t seed,
    Column a,
    Column b,
    size_type width
):
    """
    Returns the minhash values for each string.
    This function uses MurmurHash3_x64_128 for the hash algorithm.

    For details, see :cpp:func:`minhash64`.

    Parameters
    ----------
    input : Column
        Strings column to compute minhash
    seed : uint64_t
        Seed used for the hash function
    a : Column
        1st parameter value used for the minhash algorithm.
    b : Column
        2nd parameter value used for the minhash algorithm.
    width : size_type
        Character width used for apply substrings;

    Returns
    -------
    Column
        List column of minhash values for each string per seed
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_minhash64(
            input.view(),
            seed,
            a.view(),
            b.view(),
            width
        )

    return Column.from_libcudf(move(c_result))

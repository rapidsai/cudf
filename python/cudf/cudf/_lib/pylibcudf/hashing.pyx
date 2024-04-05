# Copyright (c) 2024, NVIDIA CORPORATION.
from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.hash cimport (
    md5 as cpp_md5,
    murmurhash3_x64_128 as cpp_murmurhash3_x64_128,
    murmurhash3_x86_32 as cpp_murmurhash3_x86_32,
    sha1 as cpp_sha1,
    sha224 as cpp_sha224,
    sha256 as cpp_sha256,
    sha384 as cpp_sha384,
    sha512 as cpp_sha512,
    xxhash_64 as cpp_xxhash_64,
)
from cudf._lib.cpp.table.table cimport table

from .column cimport Column
from .table cimport Table


cpdef Column murmurhash3_x86_32(
    Table input,
    uint32_t seed
):
    """Computes the MurmurHash3 32-bit hash value of each row in the given table.

    For details, see :cpp:func:murmurhash3_x86_32.

    Parameters
    ----------

    input : Table
        The table of columns to hash
    seed : uint32_t
        Optional seed value to use for the hash function

    Returns
    -------

    pylibcudf.Column
        A column where each row is the hash of a row from the input
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_murmurhash3_x86_32(
                input.view(),
                seed
            )
        )

    return Column.from_libcudf(move(c_result))


cpdef Table murmurhash3_x64_128(
    Table input,
    uint64_t seed
):
    """Computes the MurmurHash3 64-bit hash value of each row in the given table.

    For details, see :cpp:func:murmurhash3_x64_128.

    Parameters
    ----------

    input : Table
        The table of columns to hash
    seed : uint32_t
        Optional seed value to use for the hash function

    Returns
    -------

    pylibcudf.Table
        A table of two UINT64 columns
    """
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_murmurhash3_x64_128(
                input.view(),
                seed
            )
        )

    return Table.from_libcudf(move(c_result))


cpdef Column xxhash_64(
    Table input,
    uint64_t seed
):
    cdef unique_ptr[column] c_result
    with  nogil:
        c_result = move(
            cpp_xxhash_64(
                input.view(),
                seed
            )
        )

    return Column.from_libcudf(move(c_result))


cpdef Column md5(Table input):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_md5(input.view()))
    return Column.from_libcudf(move(c_result))

cpdef Column sha1(Table input):
    """Computes the SHA-1 hash value of each row in the given table.

    For details, see :cpp:func:sha1.

    Parameters
    ----------

    input : Table
        The table of columns to hash

    Returns
    -------

    pylibcudf.Column
        A column where each row is the hash of a row from the input
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_sha1(input.view()))
    return Column.from_libcudf(move(c_result))


cpdef Column sha224(Table input):
    """Computes the SHA-224 hash value of each row in the given table.

    For details, see :cpp:func:sha224.

    Parameters
    ----------

    input : Table
        The table of columns to hash

    Returns
    -------
    pylibcudf.Column
        A column where each row is the hash of a row from the input
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_sha224(input.view()))
    return Column.from_libcudf(move(c_result))


cpdef Column sha256(Table input):
    """Computes the SHA-256 hash value of each row in the given table.

    For details, see :cpp:func:256.

    Parameters
    ----------

    input : Table
        The table of columns to hash

    Returns
    -------
    pylibcudf.Column
        A column where each row is the hash of a row from the input
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_sha256(input.view()))
    return Column.from_libcudf(move(c_result))


cpdef Column sha384(Table input):
    """Computes the SHA-384 hash value of each row in the given table.

    For details, see :cpp:func:sha384.

    Parameters
    ----------

    input : Table
        The table of columns to hash

    Returns
    -------
    pylibcudf.Column
        A column where each row is the hash of a row from the input
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_sha384(input.view()))
    return Column.from_libcudf(move(c_result))


cpdef Column sha512(Table input):
    """Computes the SHA-512 hash value of each row in the given table.

    For details, see :cpp:func:sha512.

    Parameters
    ----------

    input : Table
        The table of columns to hash

    Returns
    -------
    pylibcudf.Column
        A column where each row is the hash of a row from the input
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_sha512(input.view()))
    return Column.from_libcudf(move(c_result))

# Copyright (c) 2021, NVIDIA CORPORATION.

from cudf.utils import cudautils
from numba import types


def setup_function():
    cudautils._udf_code_cache.clear()


def assert_cache_size(size):
    assert cudautils._udf_code_cache.currsize == size


def test_first_compile_sets_cache_entry():
    # The first compilation should put an entry in the cache
    cudautils.compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)


def test_code_cache_same_code_different_function_hit():
    # Compilation of a distinct function with the same code and signature
    # should reuse the cached entry

    cudautils.compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)

    cudautils.compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)


def test_code_cache_different_types_miss():
    # Compilation of a distinct function with the same code but different types
    # should create an additional cache entry

    cudautils.compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)

    cudautils.compile_udf(lambda x: x + 1, (types.float64,))
    assert_cache_size(2)


def test_code_cache_different_cvars_miss():
    # Compilation of a distinct function with the same types and code as an
    # existing entry but different closure variables should create an
    # additional cache entry

    def gen_closure(y):
        return lambda x: x + y

    cudautils.compile_udf(gen_closure(1), (types.float32,))
    assert_cache_size(1)

    cudautils.compile_udf(gen_closure(2), (types.float32,))
    assert_cache_size(2)


def test_lambda_in_loop_code_cached():
    # Compiling a UDF defined in a loop should result in the code cache being
    # reused for each loop iteration after the first. We check for this by
    # ensuring that there is only one entry in the code cache after the loop.

    for i in range(3):
        cudautils.compile_udf(lambda x: x + 1, (types.float32,))

    assert_cache_size(1)

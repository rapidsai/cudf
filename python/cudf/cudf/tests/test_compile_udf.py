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


def test_same_function_and_types():
    # Compilation of the same function with the same types should not hit the
    # UDF code cache, but instead use the LRU cache wrapping the compile_udf
    # function. We test for this by compiling the function for:
    #
    #    1. A float32 argument
    #    2. A float64 argument
    #    3. A float32 argument
    #
    # then popping the least-recently used item. The LRU item should have a
    # float32 argument, as the last type the code cache saw had a float64
    # argument - the float32 argument for compilation 3 should have been
    # serviced by the LRU cache wrapping the compile_udf function, not the code
    # cache.
    def f(x):
        return x + 1

    cudautils.compile_udf(f, (types.float32,))
    cudautils.compile_udf(f, (types.float64,))
    cudautils.compile_udf(f, (types.float32,))

    k, v = cudautils._udf_code_cache.popitem()
    # First element of the key is the type signature, then get the first (only)
    # argument
    argtype = k[0][0]

    assert argtype == types.float32


def test_lambda_in_loop_code_cached():
    # Compiling a UDF defined in a loop should result in the code cache being
    # reused for each loop iteration after the first. We check for this by
    # ensuring that there is only one entry in the code cache after the loop.
    # We expect the LRU cache wrapping compile_udf to contain three entries,
    # because it doesn't recognize that distinct functions with the same code,
    # closure variables and type are the same.

    # Clear the LRU cache to ensure we know how many entries it should have
    cudautils.compile_udf.cache_clear()

    for i in range(3):
        cudautils.compile_udf(lambda x: x + 1, (types.float32,))

    assert_cache_size(1)
    assert cudautils.compile_udf.cache_info().currsize == 3

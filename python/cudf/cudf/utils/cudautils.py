# Copyright (c) 2018-2023, NVIDIA CORPORATION.

from pickle import dumps

import cachetools
from numba import cuda
from numba.np import numpy_support

from cudf.utils._numba import _CUDFNumbaConfig

#
# Misc kernels
#


@cuda.jit
def gpu_window_sizes_from_offset(arr, window_sizes, offset):
    i = cuda.grid(1)
    j = i
    if i < arr.size:
        while j > -1:
            if (arr[i] - arr[j]) >= offset:
                break
            j -= 1
        window_sizes[i] = i - j


def window_sizes_from_offset(arr, offset):
    window_sizes = cuda.device_array(shape=(arr.shape), dtype="int32")
    if arr.size > 0:
        with _CUDFNumbaConfig():
            gpu_window_sizes_from_offset.forall(arr.size)(
                arr, window_sizes, offset
            )
    return window_sizes


@cuda.jit
def gpu_grouped_window_sizes_from_offset(
    arr, window_sizes, group_starts, offset
):
    i = cuda.grid(1)
    j = i
    if i < arr.size:
        while j > (group_starts[i] - 1):
            if (arr[i] - arr[j]) >= offset:
                break
            j -= 1
        window_sizes[i] = i - j


def grouped_window_sizes_from_offset(arr, group_starts, offset):
    window_sizes = cuda.device_array(shape=(arr.shape), dtype="int32")
    if arr.size > 0:
        with _CUDFNumbaConfig():
            gpu_grouped_window_sizes_from_offset.forall(arr.size)(
                arr, window_sizes, group_starts, offset
            )
    return window_sizes


# This cache is keyed on the (signature, code, closure variables) of UDFs, so
# it can hit for distinct functions that are similar. The lru_cache wrapping
# compile_udf misses for these similar functions, but doesn't need to serialize
# closure variables to check for a hit.
_udf_code_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)


def make_cache_key(udf, sig):
    """
    Build a cache key for a user defined function. Used to avoid
    recompiling the same function for the same set of types
    """
    codebytes = udf.__code__.co_code
    constants = udf.__code__.co_consts
    names = udf.__code__.co_names

    if udf.__closure__ is not None:
        cvars = tuple(x.cell_contents for x in udf.__closure__)
        cvarbytes = dumps(cvars)
    else:
        cvarbytes = b""

    return names, constants, codebytes, cvarbytes, sig


def compile_udf(udf, type_signature):
    """Compile ``udf`` with `numba`

    Compile a python callable function ``udf`` with
    `numba.cuda.compile_ptx_for_current_device(device=True)` using
    ``type_signature`` into CUDA PTX together with the generated output type.

    The output is expected to be passed to the PTX parser in `libcudf`
    to generate a CUDA device function to be inlined into CUDA kernels,
    compiled at runtime and launched.

    Parameters
    ----------
    udf:
      a python callable function

    type_signature:
      a tuple that specifies types of each of the input parameters of ``udf``.
      The types should be one in `numba.types` and could be converted from
      numpy types with `numba.numpy_support.from_dtype(...)`.

    Returns
    -------
    ptx_code:
      The compiled CUDA PTX

    output_type:
      An numpy type

    """
    import cudf.core.udf

    key = make_cache_key(udf, type_signature)
    res = _udf_code_cache.get(key)
    if res:
        return res

    # We haven't compiled a function like this before, so need to fall back to
    # compilation with Numba
    ptx_code, return_type = cuda.compile_ptx_for_current_device(
        udf, type_signature, device=True
    )
    if not isinstance(return_type, cudf.core.udf.masked_typing.MaskedType):
        output_type = numpy_support.as_dtype(return_type).type
    else:
        output_type = return_type

    # Populate the cache for this function
    res = (ptx_code, output_type)
    _udf_code_cache[key] = res

    return res

# Copyright (c) 2018, NVIDIA CORPORATION.
from functools import lru_cache

import cupy
import numpy as np
from numba import cuda

import cudf
from cudf.utils.utils import check_equals_float, check_equals_int, rint

try:
    # Numba >= 0.49
    from numba.np import numpy_support
except ImportError:
    # Numba <= 0.49
    from numba import numpy_support


# GPU array type casting


def as_contiguous(arr):
    assert arr.ndim == 1
    cupy_dtype = arr.dtype
    if np.issubdtype(cupy_dtype, np.datetime64):
        cupy_dtype = np.dtype("int64")
        arr = arr.view("int64")
    out = cupy.ascontiguousarray(cupy.asarray(arr))
    return cuda.as_cuda_array(out).view(arr.dtype)


# Mask utils


def full(size, value, dtype):
    cupy_dtype = dtype
    if np.issubdtype(cupy_dtype, np.datetime64):
        time_unit, _ = np.datetime_data(cupy_dtype)
        cupy_dtype = np.int64
        value = np.datetime64(value, time_unit).view(cupy_dtype)

    out = cupy.full(size, value, cupy_dtype)
    return cuda.as_cuda_array(out).view(dtype)


#
# Misc kernels
#


@cuda.jit
def gpu_diff(in_col, out_col, out_mask, N):
    """Calculate the difference between values at positions i and i - N in an
    array and store the output in a new array.
    """
    i = cuda.grid(1)

    if N > 0:
        if i < in_col.size:
            out_col[i] = in_col[i] - in_col[i - N]
            out_mask[i] = True
        if i < N:
            out_mask[i] = False
    else:
        if i <= (in_col.size + N):
            out_col[i] = in_col[i] - in_col[i - N]
            out_mask[i] = True
        if i >= (in_col.size + N) and i < in_col.size:
            out_mask[i] = False


@cuda.jit
def gpu_round(in_col, out_col, decimal):
    i = cuda.grid(1)
    f = 10 ** decimal

    if i < in_col.size:
        ret = in_col[i] * f
        ret = rint(ret)
        tmp = ret / f
        out_col[i] = tmp


def apply_round(data, decimal):
    output_dary = cuda.device_array_like(data)
    if output_dary.size > 0:
        gpu_round.forall(output_dary.size)(data, output_dary, decimal)
    return output_dary


# Find segments


@cuda.jit
def gpu_mark_found_int(arr, val, out, not_found):
    i = cuda.grid(1)
    if i < arr.size:
        if check_equals_int(arr[i], val):
            out[i] = i
        else:
            out[i] = not_found


@cuda.jit
def gpu_mark_found_float(arr, val, out, not_found):
    i = cuda.grid(1)
    if i < arr.size:
        # TODO: Remove val typecast to float(val)
        # once numba minimum version is pinned
        # at 0.51.1, this will have a very slight
        # performance improvement. Related
        # discussion in : https://github.com/rapidsai/cudf/pull/6073
        if check_equals_float(arr[i], float(val)):
            out[i] = i
        else:
            out[i] = not_found


@cuda.jit
def gpu_mark_gt(arr, val, out, not_found):
    i = cuda.grid(1)
    if i < arr.size:
        if arr[i] > val:
            out[i] = i
        else:
            out[i] = not_found


@cuda.jit
def gpu_mark_lt(arr, val, out, not_found):
    i = cuda.grid(1)
    if i < arr.size:
        if arr[i] < val:
            out[i] = i
        else:
            out[i] = not_found


def find_index_of_val(arr, val, mask=None, compare="eq"):
    """
    Returns the indices of the occurrence of *val* in *arr*
    as per *compare*, if not found it will be filled with
    size of *arr*

    Parameters
    ----------
    arr : device array
    val : scalar
    mask : mask of the array
    compare: str ('gt', 'lt', or 'eq' (default))
    """
    found = cuda.device_array(shape=(arr.shape), dtype="int32")
    if found.size > 0:
        if compare == "gt":
            gpu_mark_gt.forall(found.size)(arr, val, found, arr.size)
        elif compare == "lt":
            gpu_mark_lt.forall(found.size)(arr, val, found, arr.size)
        else:
            if arr.dtype in ("float32", "float64"):
                gpu_mark_found_float.forall(found.size)(
                    arr, val, found, arr.size
                )
            else:
                gpu_mark_found_int.forall(found.size)(
                    arr, val, found, arr.size
                )

    return cudf.core.column.column.as_column(found).set_mask(mask)


def find_first(arr, val, mask=None, compare="eq"):
    """
    Returns the index of the first occurrence of *val* in *arr*..
    Or the first occurrence of *arr* *compare* *val*, if *compare* is not eq
    Otherwise, returns -1.

    Parameters
    ----------
    arr : device array
    val : scalar
    mask : mask of the array
    compare: str ('gt', 'lt', or 'eq' (default))
    """

    found_col = find_index_of_val(arr, val, mask=mask, compare=compare)
    found_col = found_col.find_and_replace([arr.size], [None], True)

    min_index = found_col.min()
    return -1 if min_index is None or np.isnan(min_index) else min_index


def find_last(arr, val, mask=None, compare="eq"):
    """
    Returns the index of the last occurrence of *val* in *arr*.
    Or the last occurrence of *arr* *compare* *val*, if *compare* is not eq
    Otherwise, returns -1.

    Parameters
    ----------
    arr : device array
    val : scalar
    mask : mask of the array
    compare: str ('gt', 'lt', or 'eq' (default))
    """

    found_col = find_index_of_val(arr, val, mask=mask, compare=compare)
    found_col = found_col.find_and_replace([arr.size], [None], True)

    max_index = found_col.max()
    return -1 if max_index is None or np.isnan(max_index) else max_index


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
        gpu_grouped_window_sizes_from_offset.forall(arr.size)(
            arr, window_sizes, group_starts, offset
        )
    return window_sizes


@lru_cache(maxsize=32)
def compile_udf(udf, type_signature):
    """Compile ``udf`` with `numba`

    Compile a python callable function ``udf`` with
    `numba.cuda.compile_ptx_for_current_device(device=True)` using
    ``type_signature`` into CUDA PTX together with the generated output type.

    The output is expected to be passed to the PTX parser in `libcudf`
    to generate a CUDA device function to be inlined into CUDA kernels,
    compiled at runtime and launched.

    Parameters
    --------
    udf:
      a python callable function

    type_signature:
      a tuple that specifies types of each of the input parameters of ``udf``.
      The types should be one in `numba.types` and could be converted from
      numpy types with `numba.numpy_support.from_dtype(...)`.

    Returns
    --------
    ptx_code:
      The compiled CUDA PTX

    output_type:
      An numpy type

    """
    ptx_code, return_type = cuda.compile_ptx_for_current_device(
        udf, type_signature, device=True
    )
    output_type = numpy_support.as_dtype(return_type)
    return (ptx_code, output_type.type)

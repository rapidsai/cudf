# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from typing import Any, Callable, Dict, List

import cachetools
import cupy as cp
import numpy as np
from numba import cuda, typeof
from numba.core.errors import TypingError
from numba.np import numpy_support
from numba.types import CPointer, Poison, Tuple, boolean, int64, void

import rmm

from cudf.core.column.column import as_column
from cudf.core.dtypes import CategoricalDtype
from cudf.core.udf.masked_typing import MaskedType
from cudf.utils import cudautils
from cudf.utils.dtypes import (
    BOOL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
)
from cudf.utils.utils import _cudf_nvtx_annotate

JIT_SUPPORTED_TYPES = (
    NUMERIC_TYPES | BOOL_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES
)
libcudf_bitmask_type = numpy_support.from_dtype(np.dtype("int32"))
MASK_BITSIZE = np.dtype("int32").itemsize * 8

precompiled: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)
arg_handlers: List[Any] = []
ptx_files: List[Any] = []
masked_array_types: Dict[Any, Any] = {}
launch_arg_getters: Dict[Any, Any] = {}
output_col_getters: Dict[Any, Any] = {}


@_cudf_nvtx_annotate
def _get_udf_return_type(argty, func: Callable, args=()):
    """
    Get the return type of a masked UDF for a given set of argument dtypes. It
    is assumed that the function consumes a dictionary whose keys are strings
    and whose values are of MaskedType. Initially assume that the UDF may be
    written to utilize any field in the row - including those containing an
    unsupported dtype. If an unsupported dtype is actually used in the function
    the compilation should fail at `compile_udf`. If compilation succeeds, one
    can infer that the function does not use any of the columns of unsupported
    dtype - meaning we can drop them going forward and the UDF will still end
    up getting fed rows containing all the fields it actually needs to use to
    compute the answer for that row.
    """

    # present a row containing all fields to the UDF and try and compile
    compile_sig = (argty, *(typeof(arg) for arg in args))

    # Get the return type. The PTX is also returned by compile_udf, but is not
    # needed here.
    ptx, output_type = cudautils.compile_udf(func, compile_sig)

    if not isinstance(output_type, MaskedType):
        numba_output_type = numpy_support.from_dtype(np.dtype(output_type))
    else:
        numba_output_type = output_type

    result = (
        numba_output_type
        if not isinstance(numba_output_type, MaskedType)
        else numba_output_type.value_type
    )
    result = result if result.is_internal else result.return_type

    # _get_udf_return_type will throw a TypingError if the user tries to use
    # a field in the row containing an unsupported dtype, except in the
    # edge case where all the function does is return that element:

    # def f(row):
    #    return row[<bad dtype key>]
    # In this case numba is happy to return MaskedType(<bad dtype key>)
    # because it relies on not finding overloaded operators for types to raise
    # the exception, so we have to explicitly check for that case.
    if isinstance(result, Poison):
        raise TypingError(str(result))

    return result


def _is_jit_supported_type(dtype):
    # category dtype isn't hashable
    if isinstance(dtype, CategoricalDtype):
        return False
    return str(dtype) in JIT_SUPPORTED_TYPES


def _all_dtypes_from_frame(frame):
    return {
        colname: col.dtype
        if _is_jit_supported_type(col.dtype)
        else np.dtype("O")
        for colname, col in frame._data.items()
    }


def _supported_dtypes_from_frame(frame):
    return {
        colname: col.dtype
        for colname, col in frame._data.items()
        if _is_jit_supported_type(col.dtype)
    }


def _supported_cols_from_frame(frame):
    return {
        colname: col
        for colname, col in frame._data.items()
        if _is_jit_supported_type(col.dtype)
    }


def _masked_array_type_from_col(col):
    """
    Return a type representing a tuple of arrays,
    the first element an array of the numba type
    corresponding to `dtype`, and the second an
    array of bools representing a mask.
    """

    col_type = masked_array_types.get(col.dtype)
    if col_type:
        col_type = CPointer(col_type)
    else:
        nb_scalar_ty = numpy_support.from_dtype(col.dtype)
        col_type = nb_scalar_ty[::1]

    if col.mask is None:
        return col_type
    else:
        return Tuple((col_type, libcudf_bitmask_type[::1]))


def _construct_signature(frame, return_type, args):
    """
    Build the signature of numba types that will be used to
    actually JIT the kernel itself later, accounting for types
    and offsets. Skips columns with unsupported dtypes.
    """
    if not return_type.is_internal:
        return_type = CPointer(return_type)
    else:
        return_type = return_type[::1]
    # Tuple of arrays, first the output data array, then the mask
    return_type = Tuple((return_type, boolean[::1]))
    offsets = []
    sig = [return_type, int64]
    for col in _supported_cols_from_frame(frame).values():
        sig.append(_masked_array_type_from_col(col))
        offsets.append(int64)

    # return_type, size, data, masks, offsets, extra args
    sig = void(*(sig + offsets + [typeof(arg) for arg in args]))

    return sig


@cuda.jit(device=True)
def _mask_get(mask, pos):
    """Return the validity of mask[pos] as a word."""
    return (mask[pos // MASK_BITSIZE] >> (pos % MASK_BITSIZE)) & 1


def _generate_cache_key(frame, func: Callable):
    """Create a cache key that uniquely identifies a compilation.

    A new compilation is needed any time any of the following things change:
    - The UDF itself as defined in python by the user
    - The types of the columns utilized by the UDF
    - The existence of the input columns masks
    """
    return (
        *cudautils.make_cache_key(
            func, tuple(_all_dtypes_from_frame(frame).values())
        ),
        *(col.mask is None for col in frame._data.values()),
        *frame._data.keys(),
    )


@_cudf_nvtx_annotate
def _compile_or_get(frame, func, args, kernel_getter=None):
    """
    Return a compiled kernel in terms of MaskedTypes that launches a
    kernel equivalent of `f` for the dtypes of `df`. The kernel uses
    a thread for each row and calls `f` using that rows data / mask
    to produce an output value and output validity for each row.

    If the UDF has already been compiled for this requested dtypes,
    a cached version will be returned instead of running compilation.

    CUDA kernels are void and do not return values. Thus, we need to
    preallocate a column of the correct dtype and pass it in as one of
    the kernel arguments. This creates a chicken-and-egg problem where
    we need the column type to compile the kernel, but normally we would
    be getting that type FROM compiling the kernel (and letting numba
    determine it as a return value). As a workaround, we compile the UDF
    itself outside the final kernel to invoke a full typing pass, which
    unfortunately is difficult to do without running full compilation.
    we then obtain the return type from that separate compilation and
    use it to allocate an output column of the right dtype.
    """

    # check to see if we already compiled this function
    cache_key = _generate_cache_key(frame, func)
    if precompiled.get(cache_key) is not None:
        kernel, masked_or_scalar = precompiled[cache_key]
        return kernel, masked_or_scalar

    # precompile the user udf to get the right return type.
    # could be a MaskedType or a scalar type.

    kernel, scalar_return_type = kernel_getter(frame, func, args)
    np_return_type = (
        numpy_support.as_dtype(scalar_return_type)
        if scalar_return_type.is_internal
        else scalar_return_type.np_dtype
    )

    precompiled[cache_key] = (kernel, np_return_type)

    return kernel, np_return_type


def _get_kernel(kernel_string, globals_, sig, func):
    """Template kernel compilation helper function."""
    f_ = cuda.jit(device=True)(func)
    globals_["f_"] = f_
    exec(kernel_string, globals_)
    _kernel = globals_["_kernel"]
    kernel = cuda.jit(sig, link=ptx_files, extensions=arg_handlers)(_kernel)

    return kernel


def _get_input_args_from_frame(fr):
    args = []
    offsets = []
    for col in _supported_cols_from_frame(fr).values():
        getter = launch_arg_getters.get(col.dtype)
        if getter:
            data = getter(col)
        else:
            data = col.data
        if col.mask is not None:
            # argument is a tuple of data, mask
            args.append((data, col.mask))
        else:
            # argument is just the data pointer
            args.append(data)
        offsets.append(col.offset)

    return args + offsets


def _return_arr_from_dtype(dt, size):
    if extensionty := masked_array_types.get(dt):
        return rmm.DeviceBuffer(size=size * extensionty.return_type.size_bytes)
    return cp.empty(size, dtype=dt)


def _post_process_output_col(col, retty):
    if getter := output_col_getters.get(retty):
        col = getter(col)
    return as_column(col, retty)

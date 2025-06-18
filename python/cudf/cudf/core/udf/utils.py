# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Any

import cachetools
import cupy as cp
import llvmlite.binding as ll
import numpy as np
from cuda.bindings import runtime
from numba import cuda, typeof
from numba.core.datamodel import default_manager, models
from numba.core.extending import register_model
from numba.np import numpy_support
from numba.types import CPointer, Record, Tuple

import rmm

from cudf._lib import strings_udf
from cudf.core.column.column import ColumnBase, as_column
from cudf.core.dtypes import dtype
from cudf.core.udf.strings_typing import (
    string_view,
    udf_string,
)
from cudf.utils import cudautils
from cudf.utils._numba import _get_ptx_file
from cudf.utils.dtypes import (
    BOOL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    STRING_TYPES,
    TIMEDELTA_TYPES,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pylibcudf as plc

    from cudf.core.buffer.buffer import Buffer
    from cudf.core.indexed_frame import IndexedFrame

# Maximum size of a string column is 2 GiB
_STRINGS_UDF_DEFAULT_HEAP_SIZE = os.environ.get("STRINGS_UDF_HEAP_SIZE", 2**31)
_heap_size = 0
_cudf_str_dtype = dtype(str)


JIT_SUPPORTED_TYPES = (
    NUMERIC_TYPES
    | BOOL_TYPES
    | DATETIME_TYPES
    | TIMEDELTA_TYPES
    | STRING_TYPES
)
libcudf_bitmask_type = numpy_support.from_dtype(np.dtype("int32"))
MASK_BITSIZE = np.dtype("int32").itemsize * 8

precompiled: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)
launch_arg_getters: dict[Any, Any] = {}


@functools.cache
def _ptx_file():
    return _get_ptx_file(
        os.path.join(
            os.path.dirname(strings_udf.__file__), "..", "core", "udf"
        ),
        "shim_",
    )


def _all_dtypes_from_frame(frame, supported_types=JIT_SUPPORTED_TYPES):
    return {
        colname: dtype if str(dtype) in supported_types else np.dtype("O")
        for colname, dtype in frame._dtypes
    }


def _supported_dtypes_from_frame(frame, supported_types=JIT_SUPPORTED_TYPES):
    return {
        colname: dtype
        for colname, dtype in frame._dtypes
        if str(dtype) in supported_types
    }


def _supported_cols_from_frame(frame, supported_types=JIT_SUPPORTED_TYPES):
    return {
        colname: col
        for colname, col in frame._column_labels_and_values
        if str(col.dtype) in supported_types
    }


def _masked_array_type_from_col(col):
    """
    Return a type representing a tuple of arrays,
    the first element an array of the numba type
    corresponding to `dtype`, and the second an
    array of bools representing a mask.
    """

    if col.dtype == _cudf_str_dtype:
        col_type = CPointer(string_view)
    else:
        nb_scalar_ty = numpy_support.from_dtype(col.dtype)
        col_type = nb_scalar_ty[::1]

    if col.mask is None:
        return col_type
    else:
        return Tuple((col_type, libcudf_bitmask_type[::1]))


class Row(Record):
    # Numba's Record type provides a convenient abstraction for representing a
    # row, in that it provides a mapping from strings (column / field names) to
    # types. However, it cannot be used directly since it assumes that all its
    # fields can be converted to NumPy types by Numba's internal conversion
    # mechanism (`numba.np_support.as_dtype). This is not the case for cuDF
    # extension types that might be the column types (e.g. masked types, string
    # types or group types).
    #
    # We use this type for type inference and type checking, but not in code
    # generation. For this use case, it is sufficient to provide a dtype for a
    # row that corresponds to any Python object.
    @property
    def dtype(self):
        return np.dtype("object")


register_model(Row)(models.RecordModel)


@cuda.jit(device=True)
def _mask_get(mask, pos):
    """Return the validity of mask[pos] as a word."""
    return (mask[pos // MASK_BITSIZE] >> (pos % MASK_BITSIZE)) & 1


def _generate_cache_key(frame, func: Callable, args, suffix="__APPLY_UDF"):
    """Create a cache key that uniquely identifies a compilation.

    A new compilation is needed any time any of the following things change:
    - The UDF itself as defined in python by the user
    - The types of the columns utilized by the UDF
    - The existence of the input columns masks
    """
    scalar_argtypes = tuple(typeof(arg) for arg in args)
    return (
        *cudautils.make_cache_key(
            func, tuple(_all_dtypes_from_frame(frame).values())
        ),
        *(col.mask is None for col in frame._columns),
        *frame._column_names,
        scalar_argtypes,
        suffix,
    )


def _get_input_args_from_frame(fr: IndexedFrame) -> list:
    args: list[Buffer | tuple[Buffer, Buffer]] = []
    offsets = []
    for col in _supported_cols_from_frame(fr).values():
        if col.dtype == _cudf_str_dtype:
            data = column_to_string_view_array_init_heap(
                col.to_pylibcudf(mode="read")
            )
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


def _return_arr_from_dtype(dtype, size):
    if dtype == _cudf_str_dtype:
        return rmm.DeviceBuffer(size=size * _get_extensionty_size(udf_string))
    return cp.empty(size, dtype=dtype)


def _post_process_output_col(col, retty):
    if retty == _cudf_str_dtype:
        return ColumnBase.from_pylibcudf(
            strings_udf.column_from_udf_string_array(col)
        )
    return as_column(col, retty)


# The only supported data layout in NVVM.
# See: https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html?#data-layout
_nvvm_data_layout = (
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
    "i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
    "v64:64:64-v128:128:128-n16:32:64"
)


def _get_extensionty_size(ty):
    """
    Return the size of an extension type in bytes
    """
    target_data = ll.create_target_data(_nvvm_data_layout)
    llty = default_manager[ty].get_value_type()
    return llty.get_abi_size(target_data)


def initfunc(f):
    """
    Decorator for initialization functions that should
    be run exactly once.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if wrapper.initialized:
            return
        wrapper.initialized = True
        return f(*args, **kwargs)

    wrapper.initialized = False
    return wrapper


@initfunc
def set_malloc_heap_size(size=None):
    """
    Heap size control for strings_udf, size in bytes.
    """
    global _heap_size
    if size is None:
        size = _STRINGS_UDF_DEFAULT_HEAP_SIZE
    if size != _heap_size:
        (ret,) = runtime.cudaDeviceSetLimit(
            runtime.cudaLimit.cudaLimitMallocHeapSize, size
        )
        if ret.value != 0:
            raise RuntimeError("Unable to set cudaMalloc heap size")

        _heap_size = size


def column_to_string_view_array_init_heap(col: plc.Column) -> Buffer:
    # lazily allocate heap only when a string needs to be returned
    return strings_udf.column_to_string_view_array(col)


class UDFError(RuntimeError):
    pass

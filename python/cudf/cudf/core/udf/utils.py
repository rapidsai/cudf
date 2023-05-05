# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import glob
import os
from typing import Any, Callable, Dict

import cachetools
import cupy as cp
import llvmlite.binding as ll
import numpy as np
from cuda import cudart
from numba import cuda, typeof
from numba.core.datamodel import default_manager, models
from numba.core.errors import TypingError
from numba.core.extending import register_model
from numba.cuda.cudadrv.driver import Linker
from numba.np import numpy_support
from numba.types import CPointer, Poison, Record, Tuple, boolean, int64, void

import rmm

from cudf._lib.strings_udf import (
    column_from_udf_string_array,
    column_to_string_view_array,
)
from cudf.api.types import is_scalar
from cudf.core.column.column import as_column
from cudf.core.dtypes import dtype
from cudf.core.udf.masked_typing import MaskedType
from cudf.core.udf.strings_typing import (
    str_view_arg_handler,
    string_view,
    udf_string,
)
from cudf.utils import cudautils
from cudf.utils.dtypes import (
    BOOL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    STRING_TYPES,
    TIMEDELTA_TYPES,
)
from cudf.utils.utils import _cudf_nvtx_annotate, initfunc

# Maximum size of a string column is 2 GiB
_STRINGS_UDF_DEFAULT_HEAP_SIZE = os.environ.get(
    "STRINGS_UDF_HEAP_SIZE", 2**31
)
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
launch_arg_getters: Dict[Any, Any] = {}


def _get_best_ptx_file(archs, max_compute_capability):
    """
    Determine of the available PTX files which one is
    the most recent up to and including the device cc
    """
    filtered_archs = [x for x in archs if x[0] <= max_compute_capability]
    if filtered_archs:
        return max(filtered_archs, key=lambda y: y[0])
    else:
        return None


def _get_ptx_file(path, prefix):
    if "RAPIDS_NO_INITIALIZE" in os.environ:
        # cc=60 ptx is always built
        cc = int(os.environ.get("STRINGS_UDF_CC", "60"))
    else:
        dev = cuda.get_current_device()

        # Load the highest compute capability file available that is less than
        # the current device's.
        cc = int("".join(str(x) for x in dev.compute_capability))
    files = glob.glob(os.path.join(path, f"{prefix}*.ptx"))
    if len(files) == 0:
        raise RuntimeError(f"Missing PTX files for cc={cc}")
    regular_sms = []

    for f in files:
        file_name = os.path.basename(f)
        sm_number = file_name.rstrip(".ptx").lstrip(prefix)
        if sm_number.endswith("a"):
            processed_sm_number = int(sm_number.rstrip("a"))
            if processed_sm_number == cc:
                return f
        else:
            regular_sms.append((int(sm_number), f))

    regular_result = None

    if regular_sms:
        regular_result = _get_best_ptx_file(regular_sms, cc)

    if regular_result is None:
        raise RuntimeError(
            "This cuDF installation is missing the necessary PTX "
            f"files that are <={cc}."
        )
    else:
        return regular_result[1]


_PTX_FILE = _get_ptx_file(os.path.dirname(__file__), "shim_")


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


def _all_dtypes_from_frame(frame, supported_types=JIT_SUPPORTED_TYPES):
    return {
        colname: col.dtype
        if str(col.dtype) in supported_types
        else np.dtype("O")
        for colname, col in frame._data.items()
    }


def _supported_dtypes_from_frame(frame, supported_types=JIT_SUPPORTED_TYPES):
    return {
        colname: col.dtype
        for colname, col in frame._data.items()
        if str(col.dtype) in supported_types
    }


def _supported_cols_from_frame(frame, supported_types=JIT_SUPPORTED_TYPES):
    return {
        colname: col
        for colname, col in frame._data.items()
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
        *(col.mask is None for col in frame._data.values()),
        *frame._data.keys(),
        scalar_argtypes,
        suffix,
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
    if not all(is_scalar(arg) for arg in args):
        raise TypeError("only scalar valued args are supported by apply")

    # check to see if we already compiled this function
    cache_key = _generate_cache_key(frame, func, args)
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
    kernel = cuda.jit(
        sig, link=[_PTX_FILE], extensions=[str_view_arg_handler]
    )(_kernel)

    return kernel


def _get_input_args_from_frame(fr):
    args = []
    offsets = []
    for col in _supported_cols_from_frame(fr).values():
        if col.dtype == _cudf_str_dtype:
            data = column_to_string_view_array_init_heap(col)
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
        return column_from_udf_string_array(col)
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


def _get_cuda_version_from_ptx_file(path):
    """
    https://docs.nvidia.com/cuda/parallel-thread-execution/
    Each PTX module must begin with a .version
    directive specifying the PTX language version

    example header:
    //
    // Generated by NVIDIA NVVM Compiler
    //
    // Compiler Build ID: CL-31057947
    // Cuda compilation tools, release 11.6, V11.6.124
    // Based on NVVM 7.0.1
    //

    .version 7.6
    .target sm_52
    .address_size 64

    """
    with open(path) as ptx_file:
        for line in ptx_file:
            if line.startswith(".version"):
                ver_line = line
                break
        else:
            raise ValueError("Could not read CUDA version from ptx file.")
    version = ver_line.strip("\n").split(" ")[1]
    # from ptx_docs/release_notes above:
    ver_map = {
        "7.5": (11, 5),
        "7.6": (11, 6),
        "7.7": (11, 7),
        "7.8": (11, 8),
        "8.0": (12, 0),
    }

    cuda_ver = ver_map.get(version)
    if cuda_ver is None:
        raise ValueError(
            f"Could not map PTX version {version} to a CUDA version"
        )

    return cuda_ver


def _setup_numba_linker(path):
    from ptxcompiler.patch import NO_DRIVER, safe_get_versions

    from cudf.core.udf.utils import (
        _get_cuda_version_from_ptx_file,
        maybe_patch_numba_linker,
    )

    versions = safe_get_versions()
    if versions != NO_DRIVER:
        driver_version, runtime_version = versions
        ptx_toolkit_version = _get_cuda_version_from_ptx_file(path)
        maybe_patch_numba_linker(
            driver_version, runtime_version, ptx_toolkit_version
        )


def maybe_patch_numba_linker(
    driver_version, runtime_version, ptx_toolkit_version
):
    from cubinlinker.patch import (
        _numba_version_ok,
        get_logger,
        new_patched_linker,
    )

    # Numba thinks cubinlinker is only needed if the driver is older than
    # the ctk, but when PTX files are present, it might also need to patch
    # because those PTX files may newer than the driver as well
    logger = get_logger()

    if (driver_version < ptx_toolkit_version) or (
        driver_version < runtime_version
    ):
        logger.debug(
            "Driver version %s.%s needs patching due to PTX files"
            % driver_version
        )
        if _numba_version_ok:
            logger.debug("Patching Numba Linker")
            Linker.new = new_patched_linker
        else:
            logger.debug("Cannot patch Numba Linker - unsupported version")


@initfunc
def set_malloc_heap_size(size=None):
    """
    Heap size control for strings_udf, size in bytes.
    """
    global _heap_size
    if size is None:
        size = _STRINGS_UDF_DEFAULT_HEAP_SIZE
    if size != _heap_size:
        (ret,) = cudart.cudaDeviceSetLimit(
            cudart.cudaLimit.cudaLimitMallocHeapSize, size
        )
        if ret.value != 0:
            raise RuntimeError("Unable to set cudaMalloc heap size")

        _heap_size = size


def column_to_string_view_array_init_heap(col):
    # lazily allocate heap only when a string needs to be returned
    return column_to_string_view_array(col)

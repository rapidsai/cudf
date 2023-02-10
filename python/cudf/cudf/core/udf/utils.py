# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import glob
import os
from typing import Any, Callable, Dict, List

import cachetools
import cupy as cp
import llvmlite.binding as ll
import numpy as np
from cubinlinker.patch import _numba_version_ok, get_logger, new_patched_linker
from numba import cuda, typeof
from numba.core.datamodel import default_manager
from numba.core.errors import TypingError
from numba.cuda.cudadrv import nvvm
from numba.cuda.cudadrv.driver import Linker
from numba.np import numpy_support
from numba.types import CPointer, Poison, Tuple, boolean, int64, void

import rmm

from cudf.core.column.column import as_column
from cudf.core.udf.masked_typing import MaskedType
from cudf.utils import cudautils
from cudf.utils.dtypes import (
    BOOL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
)
from cudf.utils.utils import _cudf_nvtx_annotate

logger = get_logger()


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


def _get_extensionty_size(ty):
    """
    Return the size of an extension type in bytes
    """
    data_layout = nvvm.data_layout
    if isinstance(data_layout, dict):
        data_layout = data_layout[64]
    target_data = ll.create_target_data(data_layout)
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
    # Numba thinks cubinlinker is only needed if the driver is older than
    # the ctk, but when PTX files are present, it might also need to patch
    # because those PTX files may newer than the driver as well
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

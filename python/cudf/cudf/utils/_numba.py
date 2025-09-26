# Copyright (c) 2023-2025, NVIDIA CORPORATION.
from __future__ import annotations

from pickle import dumps

import cachetools
import numba
from numba import config as numba_config, cuda
from numba.np import numpy_support
from packaging import version

# This cache is keyed on the (signature, code, closure variables) of UDFs, so
# it can hit for distinct functions that are similar. The lru_cache wrapping
# compile_udf misses for these similar functions, but doesn't need to serialize
# closure variables to check for a hit.
_udf_code_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)


# Avoids using contextlib.contextmanager due to additional overhead
class _CUDFNumbaConfig:
    def __enter__(self) -> None:
        self.CUDA_LOW_OCCUPANCY_WARNINGS = (
            numba_config.CUDA_LOW_OCCUPANCY_WARNINGS
        )
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

        self.is_numba_lt_061 = version.parse(
            numba.__version__
        ) < version.parse("0.61")

        if self.is_numba_lt_061:
            self.CAPTURED_ERRORS = numba_config.CAPTURED_ERRORS
            numba_config.CAPTURED_ERRORS = "new_style"

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = (
            self.CUDA_LOW_OCCUPANCY_WARNINGS
        )
        if self.is_numba_lt_061:
            numba_config.CAPTURED_ERRORS = self.CAPTURED_ERRORS


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
    key = make_cache_key(udf, type_signature)
    res = _udf_code_cache.get(key)
    if res:
        return res

    # We haven't compiled a function like this before, so need to fall back to
    # compilation with Numba
    ptx_code, return_type = cuda.compile_ptx_for_current_device(
        udf, type_signature, device=True
    )
    if return_type.is_internal:
        output_type = numpy_support.as_dtype(return_type).type
    else:
        output_type = return_type

    # Populate the cache for this function
    res = (ptx_code, output_type)
    _udf_code_cache[key] = res

    return res

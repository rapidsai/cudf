from typing import Callable

import numpy as np
from numba import typeof
from numba.np import numpy_support
from numba.types import Tuple, boolean, int64, void
from nvtx import annotate

from cudf.core.dtypes import CategoricalDtype
from cudf.core.udf.typing import MaskedType
from cudf.utils import cudautils
from cudf.utils.dtypes import (
    BOOL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
)

JIT_SUPPORTED_TYPES = (
    NUMERIC_TYPES | BOOL_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES
)

libcudf_bitmask_type = numpy_support.from_dtype(np.dtype("int32"))


@annotate("NUMBA JIT", color="green", domain="cudf_python")
def get_udf_return_type(argty, func: Callable, args=()):

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

    return (
        numba_output_type
        if not isinstance(numba_output_type, MaskedType)
        else numba_output_type.value_type
    )


def _is_jit_supported_type(dtype):
    # category dtype isn't hashable
    if isinstance(dtype, CategoricalDtype):
        return False
    return str(dtype) in JIT_SUPPORTED_TYPES


def all_dtypes_from_frame(frame):
    return {
        colname: col.dtype
        if _is_jit_supported_type(col.dtype)
        else np.dtype("O")
        for colname, col in frame._data.items()
    }


def supported_dtypes_from_frame(frame):
    return {
        colname: col.dtype
        for colname, col in frame._data.items()
        if _is_jit_supported_type(col.dtype)
    }


def supported_cols_from_frame(frame):
    return {
        colname: col
        for colname, col in frame._data.items()
        if _is_jit_supported_type(col.dtype)
    }


def masked_array_type_from_col(col):
    """
    Return a type representing a tuple of arrays,
    the first element an array of the numba type
    corresponding to `dtype`, and the second an
    array of bools representing a mask.
    """
    nb_scalar_ty = numpy_support.from_dtype(col.dtype)
    if col.mask is None:
        return nb_scalar_ty[::1]
    else:
        return Tuple((nb_scalar_ty[::1], libcudf_bitmask_type[::1]))


def construct_signature(frame, return_type, args):
    """
    Build the signature of numba types that will be used to
    actually JIT the kernel itself later, accounting for types
    and offsets. Skips columns with unsupported dtypes.
    """

    # Tuple of arrays, first the output data array, then the mask
    return_type = Tuple((return_type[::1], boolean[::1]))
    offsets = []
    sig = [return_type, int64]
    for col in supported_cols_from_frame(frame).values():
        sig.append(masked_array_type_from_col(col))
        offsets.append(int64)

    # return_type, size, data, masks, offsets, extra args
    sig = void(*(sig + offsets + [typeof(arg) for arg in args]))

    return sig

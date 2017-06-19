"""
This file provide binding to the libgdf library.
"""
import numpy as np

from libgdf_cffi import ffi, libgdf
from . import cudautils


def columnview(size, data, mask=None, dtype=None):
    """
    Make a column view.
    """
    def unwrap(buffer):
        if buffer is None:
            return ffi.NULL
        devary = buffer.to_gpu_array()
        return ffi.cast('void*', devary.device_ctypes_pointer.value)

    dtype = dtype or data.dtype
    colview = ffi.new('gdf_column*')
    libgdf.gdf_column_view(colview, unwrap(data), unwrap(mask), size,
                           np_to_gdf_dtype(dtype))

    return colview


def apply_binaryop(binop, lhs, rhs, out):
    """Apply binary operator *binop* to operands *lhs* and *rhs*.
    The result is stored to *out*.

    Returns the number of null values.
    """
    args = (lhs._cffi_view, rhs._cffi_view, out._cffi_view)
    # apply binary operator
    binop(*args)
    # validity mask
    if out.has_null_mask:
        return apply_mask_and(lhs, rhs, out)
    else:
        return 0


def apply_unaryop(unaop, inp, out):
    """Apply unary operator *unaop* to *inp* and store to *out*.

    """
    args = (inp._cffi_view, out._cffi_view)
    # apply unary operator
    unaop(*args)


def apply_mask_and(series, mask, out):
    args = (series._cffi_view, mask._cffi_view, out._cffi_view)
    libgdf.gdf_validity_and(*args)
    nnz = cudautils.count_nonzero_mask(out._mask.mem)
    return len(out) - nnz


def np_to_gdf_dtype(dtype):
    """Util to convert numpy dtype to gdf dtype.
    """
    return {
        np.float64: libgdf.GDF_FLOAT64,
        np.float32: libgdf.GDF_FLOAT32,
        np.int64:   libgdf.GDF_INT64,
        np.int32:   libgdf.GDF_INT32,
        np.int8:    libgdf.GDF_INT8,
        np.bool_:   libgdf.GDF_INT8,
    }[np.dtype(dtype).type]


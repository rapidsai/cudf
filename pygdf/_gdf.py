"""
This file provide binding to the libgdf library.
"""
import numpy as np

from libgdf_cffi import ffi, libgdf


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
    """
    binop(lhs._cffi_view, rhs._cffi_view, out._cffi_view)


def apply_unaryop(unaop, inp, out):
    """Apply unary operator *unaop* to *inp* and store to *out*.
    """
    unaop(inp._cffi_view, out._cffi_view)


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


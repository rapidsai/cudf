"""
This file provide binding to the libgdf library.
"""
import numpy as np

from numba import cuda

from libgdf_cffi import ffi, libgdf
from . import cudautils


def unwrap_devary(devary):
    ptrval = devary.device_ctypes_pointer.value
    ptrval = ptrval or ffi.NULL   # replace None with NULL
    return ffi.cast('void*', ptrval)


def columnview(size, data, mask=None, dtype=None):
    """
    Make a column view.

    Parameters
    ----------
    size : int
        Data count.
    data : Buffer
        The data buffer.
    mask : Buffer; optional
        The mask buffer.
    dtype : numpy.dtype; optional
        The dtype of the data.  Defaults to *data.dtype*.
    """
    def unwrap(buffer):
        if buffer is None:
            return ffi.NULL
        devary = buffer.to_gpu_array()
        return unwrap_devary(devary)

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


def apply_reduce(fn, inp):
    # allocate output+temp array
    outsz = libgdf.gdf_reduce_optimal_output_size()
    out = cuda.device_array(outsz, dtype=inp.dtype)
    # call reduction
    fn(inp._cffi_view, unwrap_devary(out), outsz)
    # return 1st element
    return out[0]


def apply_sort(sr_keys, sr_vals, ascending=True):
    nelem = len(sr_keys)
    begin_bit = 0
    end_bit = sr_keys.dtype.itemsize * 8
    plan = libgdf.gdf_radixsort_plan(nelem, not ascending, begin_bit, end_bit)
    sizeof_key = sr_keys.dtype.itemsize
    sizeof_val = sr_vals.dtype.itemsize
    try:
        libgdf.gdf_radixsort_plan_setup(plan, sizeof_key, sizeof_val)
        libgdf.gdf_radixsort_generic(plan,
                                     sr_keys._cffi_view,
                                     sr_vals._cffi_view)
    finally:
        libgdf.gdf_radixsort_plan_free(plan)

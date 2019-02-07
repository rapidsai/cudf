# Copyright (c) 2018, NVIDIA CORPORATION.

"""
This file provide binding to the libgdf library.
"""
import contextlib
import itertools

import numpy as np
import pandas as pd
import pyarrow as pa

from libgdf_cffi import ffi, libgdf
from librmm_cffi import librmm as rmm

from cudf.utils import cudautils
from cudf.utils.utils import calc_chunk_size, mask_dtype, mask_bitsize


def unwrap_devary(devary):
    ptrval = devary.device_ctypes_pointer.value
    ptrval = ptrval or ffi.NULL   # replace None with NULL
    return ffi.cast('void*', ptrval)


def unwrap_mask(devary):
    ptrval = devary.device_ctypes_pointer.value
    ptrval = ptrval or ffi.NULL   # replace None with NULL
    return ffi.cast('gdf_valid_type*', ptrval), ptrval


def columnview_from_devary(devary, dtype=None):
    return _columnview(size=devary.size,  data=unwrap_devary(devary),
                       mask=ffi.NULL, dtype=dtype or devary.dtype,
                       null_count=0)


def _columnview(size, data, mask, dtype, null_count):
    colview = ffi.new('gdf_column*')
    if null_count is None:
        libgdf.gdf_column_view(
            colview,
            data,
            mask,
            size,
            np_to_gdf_dtype(dtype),
            )
    else:
        libgdf.gdf_column_view_augmented(
            colview,
            data,
            mask,
            size,
            np_to_gdf_dtype(dtype),
            null_count,
            )
    return colview


def columnview(size, data, mask=None, dtype=None, null_count=None):
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
        assert buffer.mem.is_c_contiguous(), "libGDF expects contiguous memory"
        devary = buffer.to_gpu_array()
        return unwrap_devary(devary)

    if mask is not None:
        assert null_count is not None
    dtype = dtype or data.dtype
    return _columnview(size=size, data=unwrap(data), mask=unwrap(mask),
                       dtype=dtype, null_count=null_count)


def apply_binaryop(binop, lhs, rhs, out):
    """Apply binary operator *binop* to operands *lhs* and *rhs*.
    The result is stored to *out*.

    Returns the number of null values.
    """
    args = (lhs.cffi_view, rhs.cffi_view, out.cffi_view)
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
    args = (inp.cffi_view, out.cffi_view)
    # apply unary operator
    unaop(*args)


def apply_mask_and(col, mask, out):
    args = (col.cffi_view, mask.cffi_view, out.cffi_view)
    libgdf.gdf_validity_and(*args)
    nnz = count_nonzero_mask(out.mask.mem, size=len(out))
    return len(out) - nnz


np_gdf_dict = {np.float64: libgdf.GDF_FLOAT64,
               np.float32: libgdf.GDF_FLOAT32,
               np.int64:   libgdf.GDF_INT64,
               np.int32:   libgdf.GDF_INT32,
               np.int16:   libgdf.GDF_INT16,
               np.int8:    libgdf.GDF_INT8,
               np.bool_:   libgdf.GDF_INT8,
               np.datetime64: libgdf.GDF_DATE64}


def np_to_gdf_dtype(dtype):
    """Util to convert numpy dtype to gdf dtype.
    """
    if pd.api.types.is_categorical_dtype(dtype):
        return libgdf.GDF_INT8
    else:
        return np_gdf_dict[np.dtype(dtype).type]


def gdf_to_np_dtype(dtype):
    """Util to convert gdf dtype to numpy dtype.
    """
    return np.dtype({
         libgdf.GDF_FLOAT64: np.float64,
         libgdf.GDF_FLOAT32: np.float32,
         libgdf.GDF_INT64: np.int64,
         libgdf.GDF_INT32: np.int32,
         libgdf.GDF_INT16: np.int16,
         libgdf.GDF_INT8: np.int8,
         libgdf.GDF_DATE64: np.datetime64,
         libgdf.N_GDF_TYPES: np.int32,
         libgdf.GDF_CATEGORY: np.int32,
     }[dtype])


def np_to_pa_dtype(dtype):
    """Util to convert numpy dtype to PyArrow dtype.
    """
    return {
        np.float64:     pa.float64(),
        np.float32:     pa.float32(),
        np.int64:       pa.int64(),
        np.int32:       pa.int32(),
        np.int16:       pa.int16(),
        np.int8:        pa.int8(),
        np.bool_:       pa.int8(),
        np.datetime64:  pa.date64(),
    }[np.dtype(dtype).type]


def apply_reduce(fn, inp):
    # allocate output+temp array
    outsz = libgdf.gdf_reduce_optimal_output_size()
    out = rmm.device_array(outsz, dtype=inp.dtype)
    # call reduction
    fn(inp.cffi_view, unwrap_devary(out), outsz)
    # return 1st element
    return out[0]


_join_how_api = {
    'inner': libgdf.gdf_inner_join,
    'outer': libgdf.gdf_full_join,
    'left': libgdf.gdf_left_join,
}

_join_method_api = {
    'sort': libgdf.GDF_SORT,
    'hash': libgdf.GDF_HASH
}


def cffi_view_to_column_mem(cffi_view):
    intaddr = int(ffi.cast("uintptr_t", cffi_view.data))
    data = rmm.device_array_from_ptr(intaddr,
                                     nelem=cffi_view.size,
                                     dtype=gdf_to_np_dtype(cffi_view.dtype),
                                     finalizer=rmm._make_finalizer(intaddr, 0))

    if cffi_view.valid:
        intaddr = int(ffi.cast("uintptr_t", cffi_view.valid))
        mask = rmm.device_array_from_ptr(intaddr,
                                         nelem=calc_chunk_size(cffi_view.size,
                                                               mask_bitsize),
                                         dtype=mask_dtype,
                                         finalizer=rmm._make_finalizer(intaddr,
                                                                       0))
    else:
        mask = None

    return data, mask


@contextlib.contextmanager
def apply_join(col_lhs, col_rhs, how, method='hash'):
    """Returns a tuple of the left and right joined indices as gpu arrays.
    """
    if(len(col_lhs) != len(col_rhs)):
        msg = "Unequal #columns in list 'col_lhs' and list 'col_rhs'"
        raise ValueError(msg)

    joiner = _join_how_api[how]
    method_api = _join_method_api[method]
    gdf_context = ffi.new('gdf_context*')

    if method == 'hash':
        libgdf.gdf_context_view(gdf_context, 0, method_api, 0, 0, 0)
    elif method == 'sort':
        libgdf.gdf_context_view(gdf_context, 1, method_api, 0, 0, 0)
    else:
        msg = "method not supported"
        raise ValueError(msg)

    col_result_l = columnview(0, None, dtype=np.int32)
    col_result_r = columnview(0, None, dtype=np.int32)

    if(how in ['left', 'inner']):
        list_lhs = []
        list_rhs = []
        for i in range(len(col_lhs)):
            list_lhs.append(col_lhs[i].cffi_view)
            list_rhs.append(col_rhs[i].cffi_view)

        # Call libgdf

        joiner(len(col_lhs), list_lhs, list_rhs, col_result_l,
               col_result_r, gdf_context)
    else:
        joiner(col_lhs[0].cffi_view, col_rhs[0].cffi_view, col_result_l,
               col_result_r)

    # Extract result

    left = rmm.device_array_from_ptr(ptr=col_result_l.data,
                                     nelem=col_result_l.size,
                                     dtype=np.int32)

    right = rmm.device_array_from_ptr(ptr=col_result_r.data,
                                      nelem=col_result_r.size,
                                      dtype=np.int32)

    yield(left, right)

    libgdf.gdf_column_free(col_result_l)
    libgdf.gdf_column_free(col_result_r)


def libgdf_join(col_lhs, col_rhs, on, how, method='sort'):
    joiner = _join_how_api[how]
    method_api = _join_method_api[method]
    gdf_context = ffi.new('gdf_context*')

    libgdf.gdf_context_view(gdf_context, 0, method_api, 0, 0, 0)

    if how not in ['left', 'inner', 'outer']:
        msg = "new join api only supports left or inner"
        raise ValueError(msg)

    list_lhs = []
    list_rhs = []
    result_cols = []

    result_col_names = []

    left_idx = []
    right_idx = []
    # idx = 0
    for name, col in col_lhs.items():
        list_lhs.append(col._column.cffi_view)
        if name not in on:
            result_cols.append(columnview(0, None, dtype=col._column.dtype))
            result_col_names.append(name)

    for name in on:
        result_cols.append(columnview(0, None,
                                      dtype=col_lhs[name]._column.dtype))
        result_col_names.append(name)
        left_idx.append(list(col_lhs.keys()).index(name))
        right_idx.append(list(col_rhs.keys()).index(name))

    for name, col in col_rhs.items():
        list_rhs.append(col._column.cffi_view)
        if name not in on:
            result_cols.append(columnview(0, None, dtype=col._column.dtype))
            result_col_names.append(name)

    num_cols_to_join = len(on)
    result_num_cols = len(list_lhs) + len(list_rhs) - num_cols_to_join

    joiner(list_lhs,
           len(list_lhs),
           left_idx,
           list_rhs,
           len(list_rhs),
           right_idx,
           num_cols_to_join,
           result_num_cols,
           result_cols,
           ffi.NULL,
           ffi.NULL,
           gdf_context)

    res = []
    valids = []

    for col in result_cols:

        intaddr = int(ffi.cast("uintptr_t", col.data))
        res.append(rmm.device_array_from_ptr(ptr=intaddr,
                                             nelem=col.size,
                                             dtype=gdf_to_np_dtype(col.dtype),
                                             finalizer=rmm._make_finalizer(
                                                 intaddr, 0)))
        intaddr = int(ffi.cast("uintptr_t", col.valid))
        valids.append(rmm.device_array_from_ptr(ptr=intaddr,
                                                nelem=calc_chunk_size(
                                                    col.size, mask_bitsize),
                                                dtype=mask_dtype,
                                                finalizer=rmm._make_finalizer(
                                                    intaddr, 0)))

    return res, valids


def apply_prefixsum(col_inp, col_out, inclusive):
    libgdf.gdf_prefixsum_generic(col_inp, col_out, inclusive)


def apply_segsort(col_keys, col_vals, segments, descending=False,
                  plan=None):
    """Inplace segemented sort

    Parameters
    ----------
    col_keys : Column
    col_vals : Column
    segments : device array
    """
    # prepare
    nelem = len(col_keys)
    if nelem == segments.size:
        # As many seguments as there are elements.
        # Nothing to do.
        return

    if plan is None:
        plan = SegmentedRadixortPlan(nelem, col_keys.dtype, col_vals.dtype,
                                     descending=descending)

    plan.sort(segments, col_keys, col_vals)
    return plan


class SegmentedRadixortPlan(object):
    def __init__(self, nelem, key_dtype, val_dtype, descending=False):
        begin_bit = 0
        self.sizeof_key = key_dtype.itemsize
        self.sizeof_val = val_dtype.itemsize
        end_bit = self.sizeof_key * 8
        plan = libgdf.gdf_segmented_radixsort_plan(nelem, descending,
                                                   begin_bit, end_bit)
        self.plan = plan
        self.nelem = nelem
        self.is_closed = False
        self.setup()

    def __del__(self):
        if not self.is_closed:
            self.close()

    def close(self):
        libgdf.gdf_segmented_radixsort_plan_free(self.plan)
        self.is_closed = True
        self.plan = None

    def setup(self):
        libgdf.gdf_segmented_radixsort_plan_setup(self.plan, self.sizeof_key,
                                                  self.sizeof_val)

    def sort(self, segments, col_keys, col_vals):
        seg_dtype = np.uint32
        segsize_limit = 2 ** 16 - 1

        d_fullsegs = rmm.device_array(segments.size + 1, dtype=seg_dtype)
        d_begins = d_fullsegs[:-1]
        d_ends = d_fullsegs[1:]

        # Note: .astype is required below because .copy_to_device
        #       is just a plain memcpy
        d_begins.copy_to_device(cudautils.astype(segments, dtype=seg_dtype))
        d_ends[-1:].copy_to_device(np.require([self.nelem], dtype=seg_dtype))

        # The following is to handle the segument size limit due to
        # max CUDA grid size.
        range0 = range(0, segments.size, segsize_limit)
        range1 = itertools.chain(range0[1:], [segments.size])
        for s, e in zip(range0, range1):
            segsize = e - s
            libgdf.gdf_segmented_radixsort_generic(self.plan,
                                                   col_keys.cffi_view,
                                                   col_vals.cffi_view,
                                                   segsize,
                                                   unwrap_devary(d_begins[s:]),
                                                   unwrap_devary(d_ends[s:]))


def hash_columns(columns, result):
    """Hash the *columns* and store in *result*.
    Returns *result*
    """
    assert len(columns) > 0
    assert result.dtype == np.int32
    # No-op for 0-sized
    if len(result) == 0:
        return result
    col_input = [col.cffi_view for col in columns]
    col_out = result.cffi_view
    ncols = len(col_input)
    hashfn = libgdf.GDF_HASH_MURMUR3
    libgdf.gdf_hash(ncols, col_input, hashfn, col_out)
    return result


def hash_partition(input_columns, key_indices, nparts, output_columns):
    """Partition the input_columns by the hash values on the keys.

    Parameters
    ----------
    input_columns : sequence of Column
    key_indices : sequence of int
        Indices into `input_columns` that indicates the key columns.
    nparts : int
        number of partitions

    Returns
    -------
    partition_offsets : list of int
        Each index indicates the start of a partition.
    """
    assert len(input_columns) == len(output_columns)

    col_inputs = [col.cffi_view for col in input_columns]
    col_outputs = [col.cffi_view for col in output_columns]
    offsets = ffi.new('int[]', nparts)
    hashfn = libgdf.GDF_HASH_MURMUR3

    libgdf.gdf_hash_partition(
        len(col_inputs),
        col_inputs,
        key_indices,
        len(key_indices),
        nparts,
        col_outputs,
        offsets,
        hashfn
    )

    offsets = list(offsets)
    return offsets


def _column_concat(cols_to_concat, output_col):
    col_inputs = [col.cffi_view for col in cols_to_concat]
    libgdf.gdf_column_concat(output_col.cffi_view, col_inputs, len(col_inputs))
    return output_col


def count_nonzero_mask(mask, size):
    assert mask.size * mask_bitsize >= size
    nnz = ffi.new('int*')
    nnz[0] = 0
    mask_ptr, addr = unwrap_mask(mask)

    if addr != ffi.NULL:
        libgdf.gdf_count_nonzero_mask(mask_ptr, size, nnz)

    return nnz[0]


_GDF_COLORS = {
    'green':    libgdf.GDF_GREEN,
    'blue':     libgdf.GDF_BLUE,
    'yellow':   libgdf.GDF_YELLOW,
    'purple':   libgdf.GDF_PURPLE,
    'cyan':     libgdf.GDF_CYAN,
    'red':      libgdf.GDF_RED,
    'white':    libgdf.GDF_WHITE,
    'darkgreen': libgdf.GDF_DARK_GREEN,
    'orange':   libgdf.GDF_ORANGE,
}


def str_to_gdf_color(s):
    """Util to convert str to gdf_color type.
    """
    return _GDF_COLORS[s.lower()]


def nvtx_range_push(name, color='green'):
    """
    Demarcate the beginning of a user-defined NVTX range.

    Parameters
    ----------
    name : str
        The name of the NVTX range
    color : str
        The color to use for the range.
        Can be named color or hex RGB string.
    """
    name_c = ffi.new("char[]", name.encode('ascii'))

    try:
        color = int(color, 16)  # only works if color is a hex string
        libgdf.gdf_nvtx_range_push_hex(name_c, ffi.cast('unsigned int', color))
    except ValueError:
        color = str_to_gdf_color(color)
        libgdf.gdf_nvtx_range_push(name_c, color)


def nvtx_range_pop():
    """ Demarcate the end of the inner-most range.
    """
    libgdf.gdf_nvtx_range_pop()


def rmm_initialize():
    rmm.initialize()
    return True


def rmm_finalize():
    rmm.finalize()
    return True


_GDF_QUANTILE_METHODS = {
    'linear': libgdf.GDF_QUANT_LINEAR,
    'lower': libgdf.GDF_QUANT_LOWER,
    'higher': libgdf.GDF_QUANT_HIGHER,
    'midpoint': libgdf.GDF_QUANT_MIDPOINT,
    'nearest': libgdf.GDF_QUANT_NEAREST,
}


def get_quantile_method(method):
    """Util to convert method to gdf gdf_quantile_method.
    """
    return _GDF_QUANTILE_METHODS[method]


def quantile(column, quant, method, exact):
    """ Calculate the `quant` quantile for the column
    Returns value with the quantile specified by quant
    """
    gdf_context = ffi.new('gdf_context*')
    method_api = _join_method_api['sort']
    libgdf.gdf_context_view(gdf_context, 0, method_api, 0, 0, 0)
    # libgdf.gdf_context_view(gdf_context, 0, method_api, 0)
    # px = ffi.new("double *")
    res = []
    for q in quant:
        px = ffi.new("double *")
        if exact:
            libgdf.gdf_quantile_exact(column.cffi_view,
                                      get_quantile_method(method),
                                      q,
                                      ffi.cast('void *', px),
                                      gdf_context)
        else:
            libgdf.gdf_quantile_aprrox(column.cffi_view,
                                       q,
                                       ffi.cast('void *', px),
                                       gdf_context)
        res.append(px[0])
    return res

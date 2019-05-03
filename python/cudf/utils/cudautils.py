# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np

from numba import cuda, int32, numpy_support
from math import isnan

from librmm_cffi import librmm as rmm

from cudf.utils.utils import (check_equals_int, check_equals_float,
                              mask_bitsize, mask_get, mask_set, make_mask)
import nvstrings


def optimal_block_count(minblkct):
    """Return the optimal block count for a CUDA kernel launch.
    """
    return min(16, max(1, minblkct))


def to_device(ary):
    dary, _ = rmm.auto_device(ary)
    return dary


# GPU array initializer

@cuda.jit
def gpu_arange(start, size, step, out):
    i = cuda.grid(1)
    if i < size:
        out[i] = i * step + start


def arange(start, stop=None, step=1, dtype=np.int64):
    if stop is None:
        start, stop = 0, start
    if step < 0:
        size = (stop - start + 1) // step + 1
    else:
        size = (stop - start - 1) // step + 1

    if size < 0:
        msgfmt = "size={size} in arange({start}, {stop}, {step}, {dtype})"
        raise ValueError(msgfmt.format(size=size, start=start, stop=stop,
                                       step=step, dtype=dtype))
    out = rmm.device_array(shape=int(size), dtype=dtype)
    if size > 0:
        gpu_arange.forall(size)(start, size, step, out)
    return out


@cuda.jit
def gpu_arange_reversed(size, out):
    i = cuda.grid(1)
    if i < size:
        out[i] = size - i - 1


def arange_reversed(size, dtype=np.int64):
    out = rmm.device_array(size, dtype=dtype)
    if size > 0:
        gpu_arange_reversed.forall(size)(size, out)
    return out


@cuda.jit
def gpu_ones(size, out):
    i = cuda.grid(1)
    if i < size:
        out[i] = 1


def ones(size, dtype):
    out = rmm.device_array(size, dtype=dtype)
    if size > 0:
        gpu_ones.forall(size)(size, out)
    return out


@cuda.jit
def gpu_zeros(size, out):
    i = cuda.grid(1)
    if i < size:
        out[i] = 0


def zeros(size, dtype):
    out = rmm.device_array(size, dtype=dtype)
    if size > 0:
        gpu_zeros.forall(size)(size, out)
    return out


# GPU array type casting


@cuda.jit
def gpu_copy(inp, out):
    tid = cuda.grid(1)
    if tid < inp.size:
        out[tid] = inp[tid]


def astype(ary, dtype):
    if ary.dtype == np.dtype(dtype):
        return ary
    elif (
        (ary.dtype == np.int64 or ary.dtype == np.dtype('datetime64[ms]')) and
        (dtype == np.dtype('int64') or dtype == np.dtype('datetime64[ms]'))
    ):
        return ary.view(dtype)
    elif ary.size == 0:
        return rmm.device_array(shape=ary.shape, dtype=dtype)
    else:
        out = rmm.device_array(shape=ary.shape, dtype=dtype)
        if out.size > 0:
            configured = gpu_copy.forall(out.size)
            configured(ary, out)
        return out


def copy_array(arr, out=None):
    if out is None:
        out = rmm.device_array_like(arr)

    if (arr.is_c_contiguous() and out.is_c_contiguous() and
            out.size == arr.size):
        out.copy_to_device(arr)
    else:
        if arr.size > 0:
            gpu_copy.forall(arr.size)(arr, out)
    return out


def as_contiguous(arr):
    assert arr.ndim == 1
    out = rmm.device_array(shape=arr.shape, dtype=arr.dtype)
    return copy_array(arr, out=out)


# Copy column into a matrix

@cuda.jit
def gpu_copy_column(matrix, colidx, colvals):
    tid = cuda.grid(1)
    if tid < colvals.size:
        matrix[colidx, tid] = colvals[tid]


def copy_column(matrix, colidx, colvals):
    assert matrix.shape[1] == colvals.size
    if colvals.size > 0:
        configured = gpu_copy_column.forall(colvals.size)
    configured(matrix, colidx, colvals)


# Mask utils

@cuda.jit
def gpu_set_mask_from_stride(mask, stride):
    bitsize = mask_bitsize
    tid = cuda.grid(1)
    if tid < mask.size:
        base = tid * bitsize
        val = 0
        for shift in range(bitsize):
            idx = base + shift
            bit = (idx % stride) == 0
            val |= bit << shift
        mask[tid] = val


def set_mask_from_stride(mask, stride):
    taskct = mask.size
    if taskct > 0:
        configured = gpu_set_mask_from_stride.forall(taskct)
        configured(mask, stride)


@cuda.jit
def gpu_copy_to_dense(data, mask, slots, out):
    tid = cuda.grid(1)
    if tid < data.size and mask_get(mask, tid):
        idx = slots[tid]
        out[idx] = data[tid]


@cuda.jit
def gpu_invert_value(data, out):
    tid = cuda.grid(1)
    if tid < data.size:
        out[tid] = not data[tid]


def invert_mask(arr, out):
    if arr.size > 0:
        gpu_invert_value.forall(arr.size)(arr, out)


def fill_mask(data, mask, value):
    """fill a column with the same value using a custom mask

    Parameters
    ----------
    data : device array
        data
    mask : device array
        validity mask
    value : scale
        fill value

    Returns
    -------
    device array
        mask filled column with scalar value
    """

    out = rmm.device_array_like(data)
    out.copy_to_device(data)
    if data.size > 0:
        configured = gpu_fill_masked.forall(data.size)
        configured(value, mask, out)
    return out


@cuda.jit
def gpu_fill_value(data, value):
    tid = cuda.grid(1)
    if tid < data.size:
        data[tid] = value


def fill_value(arr, value):
    """Fill *arr* with value
    """
    if arr.size > 0:
        gpu_fill_value.forall(arr.size)(arr, value)


def full(size, value, dtype):
    out = rmm.device_array(size, dtype=dtype)
    fill_value(out, value)
    return out


@cuda.jit
def gpu_expand_mask_bits(bits, out):
    """Expand each bits in bitmask *bits* into an element in out.
    This is a flexible kernel that can be launch with any number of blocks
    and threads.
    """
    for i in range(cuda.grid(1), out.size, cuda.gridsize(1)):
        if i < bits.size * mask_bitsize:
            out[i] = mask_get(bits, i)


def expand_mask_bits(size, bits):
    """Expand bit-mask into byte-mask
    """
    expanded_mask = full(size, 0, dtype=np.int32)
    numtasks = min(1024, expanded_mask.size)
    if numtasks > 0:
        gpu_expand_mask_bits.forall(numtasks)(bits, expanded_mask)
    return expanded_mask


def mask_assign_slot(size, mask):
    # expand bits into bytes
    expanded_mask = expand_mask_bits(size, mask)
    # compute prefixsum
    slots = prefixsum(expanded_mask)
    sz = int(slots[slots.size - 1])
    return slots, sz


def prefixsum(vals):
    """Compute the full prefixsum.

    Given the input of N.  The output size is N + 1.
    The first value is always 0.  The last value is the sum of *vals*.
    """

    import cudf.bindings.reduce as cpp_reduce
    from cudf.dataframe.numerical import NumericalColumn
    from cudf.dataframe.buffer import Buffer

    # Allocate output
    slots = rmm.device_array(shape=vals.size + 1,
                             dtype=vals.dtype)
    # Fill 0 to slot[0]
    gpu_fill_value[1, 1](slots[:1], 0)

    # Compute prefixsum on the mask
    in_col = NumericalColumn(data=Buffer(vals), mask=None,
                             null_count=0, dtype=vals.dtype)
    out_col = NumericalColumn(data=Buffer(slots[1:]), mask=None,
                              null_count=0, dtype=vals.dtype)
    cpp_reduce.apply_scan(in_col, out_col, 'sum', inclusive=True)
    return slots


def copy_to_dense(data, mask, out=None):
    """Copy *data* with validity bits in *mask* into *out*.

    The output array can be specified in `out`.

    Return a 2-tuple of:
    * number of non-null element
    * a dense gpu array given the data and mask gpu arrays.
    """
    slots, sz = mask_assign_slot(size=data.size, mask=mask)
    if out is None:
        # output buffer is not provided
        # allocate one
        alloc_shape = sz
        out = rmm.device_array(shape=alloc_shape, dtype=data.dtype)
    else:
        # output buffer is provided
        # check it
        if sz >= out.size:
            raise ValueError('output array too small')
    if out.size > 0:
        gpu_copy_to_dense.forall(data.size)(data, mask, slots, out)
    return (sz, out)


@cuda.jit
def gpu_compact_mask_bytes(bools, bits):
    tid = cuda.grid(1)
    base = tid * mask_bitsize
    for i in range(base, base + mask_bitsize):
        if i >= bools.size:
            break
        if bools[i]:
            mask_set(bits, i)


def compact_mask_bytes(boolbytes):
    """Convert booleans (in bytes) to a bitmask
    """
    bits = make_mask(boolbytes.size)
    if bits.size > 0:
        # Fill zero
        gpu_fill_value.forall(bits.size)(bits, 0)
        # Compact
        gpu_compact_mask_bytes.forall(bits.size)(boolbytes, bits)
    return bits


@cuda.jit
def gpu_mask_from_devary(ary, bits):
    tid = cuda.grid(1)
    base = tid * mask_bitsize
    for i in range(base, base + mask_bitsize):
        if i >= len(ary):
            break
        if not isnan(ary[i]):
            mask_set(bits, i)


def mask_from_devary(ary):
    bits = make_mask(len(ary))
    if bits.size > 0:
        gpu_fill_value.forall(bits.size)(bits, 0)
        gpu_mask_from_devary.forall(bits.size)(ary, bits)
    return bits


def make_empty_mask(size):
    bits = make_mask(size)
    if bits.size > 0:
        gpu_fill_value.forall(bits.size)(bits, 0)
    return bits

#
# Gather
#


@cuda.jit
def gpu_gather(data, index, out):
    i = cuda.grid(1)
    if i < index.size:
        idx = index[i]
        # Only do it if the index is in range
        if 0 <= idx < data.size:
            out[i] = data[idx]


def gather(data, index, out=None):
    """Perform ``out = data[index]`` on the GPU
    """
    if out is None:
        out = rmm.device_array(shape=index.size, dtype=data.dtype)
    if out.size > 0:
        gpu_gather.forall(index.size)(data, index, out)
    return out


@cuda.jit
def gpu_gather_joined_index(lkeys, rkeys, lidx, ridx, out):
    gid = cuda.grid(1)
    if gid < lidx.size:
        # Try getting from the left side first
        pos = lidx[gid]
        if pos != -1:
            # Get from left
            out[gid] = lkeys[pos]
        else:
            # Get from right
            pos = ridx[gid]
            out[gid] = rkeys[pos]


def gather_joined_index(lkeys, rkeys, lidx, ridx):
    assert lidx.size == ridx.size
    out = rmm.device_array(lidx.size, dtype=lkeys.dtype)
    if out.size > 0:
        gpu_gather_joined_index.forall(lidx.size)(lkeys, rkeys, lidx, ridx,
                                                  out)
    return out


def reverse_array(data, out=None):
    rinds = arange_reversed(data.size)
    out = gather(data=data, index=rinds, out=out)
    return out


#
# Null handling
#


@cuda.jit
def gpu_fill_masked(value, validity, out):
    tid = cuda.grid(1)
    if tid < out.size:
        valid = mask_get(validity, tid)
        if not valid:
            out[tid] = value


def fillna(data, mask, value):
    out = rmm.device_array_like(data)
    out.copy_to_device(data)
    if data.size > 0:
        configured = gpu_fill_masked.forall(data.size)
        configured(value, mask, out)
    return out


@cuda.jit
def gpu_isnull(validity, out):
    tid = cuda.grid(1)
    if tid < out.size:
        valid = mask_get(validity, tid)
        if valid:
            out[tid] = False
        else:
            out[tid] = True


def isnull_mask(data, mask):
    # necessary due to rapidsai/custrings#263
    if isinstance(data, nvstrings.nvstrings):
        output_dary = rmm.device_array(data.size(), dtype=np.bool_)
    else:
        output_dary = rmm.device_array(data.size, dtype=np.bool_)

    if output_dary.size > 0:
        gpu_isnull.forall(output_dary.size)(mask, output_dary)
    return output_dary


@cuda.jit
def gpu_notna(validity, out):
    tid = cuda.grid(1)
    if tid < out.size:
        valid = mask_get(validity, tid)
        if valid:
            out[tid] = True
        else:
            out[tid] = False


def notna_mask(data, mask):
    # necessary due to rapidsai/custrings#263
    if isinstance(data, nvstrings.nvstrings):
        output_dary = rmm.device_array(data.size(), dtype=np.bool_)
    else:
        output_dary = rmm.device_array(data.size, dtype=np.bool_)

    if output_dary.size > 0:
        gpu_notna.forall(output_dary.size)(mask, output_dary)
    return output_dary


#
# Binary kernels
#

@cuda.jit
def gpu_equal_constant_masked(arr, mask, val, out):
    i = cuda.grid(1)
    if i < out.size:
        res = (arr[i] == val) if mask_get(mask, i) else False
        out[i] = res


@cuda.jit
def gpu_equal_constant(arr, val, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (arr[i] == val)


def apply_equal_constant(arr, mask, val, dtype):
    """Compute ``arr[mask] == val``

    Parameters
    ----------
    arr : device array
        data
    mask : device array
        validity mask
    val : scalar
        value to compare against
    dtype : np.dtype
        output array dtype

    Returns
    -------
    result : device array
    """
    out = rmm.device_array(shape=arr.size, dtype=dtype)
    if out.size > 0:
        if mask is not None:
            configured = gpu_equal_constant_masked.forall(out.size)
            configured(arr, mask, val, out)
        else:
            configured = gpu_equal_constant.forall(out.size)
            configured(arr, val, out)
    return out


@cuda.jit
def gpu_scale(arr, vmin, vmax, out):
    i = cuda.grid(1)
    if i < out.size:
        val = arr[i]
        out[i] = (val - vmin) / (vmax - vmin)


def compute_scale(arr, vmin, vmax):
    out = rmm.device_array(shape=arr.size, dtype=np.float64)
    if out.size > 0:
        configured = gpu_scale.forall(out.size)
        configured(arr, vmin, vmax, out)
    return out


@cuda.jit
def gpu_label(arr, cats, enc, na_sentinel, out):
    i = cuda.grid(1)
    if i < out.size:
        val = arr[i]
        out[i] = na_sentinel
        for j in range(cats.shape[0]):
            if val == cats[j]:
                res = enc[j]
                out[i] = res
                break


def apply_label(arr, cats, dtype, na_sentinel):
    """
    Parameters
    ----------
    arr : device array
        data
    cats : device array
        Unique category value
    dtype : np.dtype
        output array dtype
    na_sentinel : int
        Value to indicate missing value
    Returns
    -------
    result : device array
    """
    encs = np.asarray(list(range(cats.size)))
    d_encs = to_device(encs)
    out = rmm.device_array(shape=arr.size, dtype=dtype)
    if out.size > 0:
        configured = gpu_label.forall(out.size)
        configured(arr, cats, d_encs, na_sentinel, out)
    return out

#
# Misc kernels
#


@cuda.jit(device=True)
def gpu_unique_set_insert(vset, sz, val):
    """
    Insert *val* into the *vset* of size *sz*.
    Returns:
        *   -1: out-of-space
        * >= 0: the new size
    """
    for i in range(sz):
        # value matches, so return current size
        if vset[i] == val:
            return sz

    if sz > 0:
        i += 1
    # insert at next available slot
    if i < vset.size:
        vset[i] = val
        return sz + 1

    # out of space
    return -1


@cuda.jit
def gpu_shift(in_col, out_col, N):
    """Shift value at index i of an input array forward by N positions and
    store the output in a new array.
    """
    i = cuda.grid(1)
    if N > 0:
        if i < in_col.size:
            out_col[i] = in_col[i-N]
        if i < N:
            out_col[i] = -1
    else:
        if i <= (in_col.size + N):
            out_col[i] = in_col[i-N]
        if i >= (in_col.size + N) and i < in_col.size:
            out_col[i] = -1


MAX_FAST_UNIQUE_K = 2 * 1024


class UniqueK(object):
    _cached_kernels = {}

    def __init__(self, dtype):
        dtype = np.dtype(dtype)
        self._kernel = self._get_kernel(dtype)

    @classmethod
    def _get_kernel(cls, dtype):
        try:
            return cls._cached_kernels[dtype]
        except KeyError:
            return cls._compile(dtype)

    @classmethod
    def _compile(cls, dtype):
        nbtype = numpy_support.from_dtype(dtype)

        @cuda.jit
        def gpu_unique_k(arr, k, out, outsz_ptr):
            """
            Note: run with small blocks.
            """
            tid = cuda.threadIdx.x
            blksz = cuda.blockDim.x
            base = 0

            # shared memory
            vset_size = 0
            sm_mem_size = MAX_FAST_UNIQUE_K
            vset = cuda.shared.array(sm_mem_size, dtype=nbtype)
            share_vset_size = cuda.shared.array(1, dtype=int32)
            share_loaded = cuda.shared.array(sm_mem_size, dtype=nbtype)
            sm_mem_size = min(k, sm_mem_size)

            while vset_size < sm_mem_size and base < arr.size:
                pos = base + tid
                valid_load = min(blksz, arr.size - base)
                # load
                if tid < valid_load:
                    share_loaded[tid] = arr[pos]
                # wait for load to complete
                cuda.syncthreads()
                # thread-0 inserts
                if tid == 0:
                    for i in range(valid_load):
                        val = share_loaded[i]
                        new_size = gpu_unique_set_insert(vset, vset_size, val)
                        if new_size >= 0:
                            vset_size = new_size
                        else:
                            vset_size = sm_mem_size + 1
                    share_vset_size[0] = vset_size
                # wait until the insert is done
                cuda.syncthreads()
                vset_size = share_vset_size[0]
                # increment
                base += blksz

            # output
            if vset_size <= sm_mem_size:
                for i in range(tid, vset_size, blksz):
                    out[i] = vset[i]
                if tid == 0:
                    outsz_ptr[0] = vset_size
            else:
                outsz_ptr[0] = -1

        # cache
        cls._cached_kernels[dtype] = gpu_unique_k
        return gpu_unique_k

    def run(self, arr, k):
        if k >= MAX_FAST_UNIQUE_K:
            raise NotImplementedError('k >= {}'.format(MAX_FAST_UNIQUE_K))
        # setup mem
        outsz_ptr = rmm.device_array(shape=1, dtype=np.intp)
        out = rmm.device_array_like(arr)
        # kernel
        self._kernel[1, 64](arr, k, out, outsz_ptr)
        # copy to host
        unique_ct = outsz_ptr.copy_to_host()[0]
        if unique_ct < 0:
            raise ValueError('too many unique value (hint: increase k)')
        else:
            hout = out.copy_to_host()
            return hout[:unique_ct]


# Find segments


@cuda.jit
def gpu_mark_segment_begins_float(arr, markers):
    i = cuda.grid(1)
    if i == 0:
        markers[0] = 1
    elif 0 < i < markers.size:
        if not markers[i]:
            markers[i] = not check_equals_float(arr[i], arr[i - 1])


@cuda.jit
def gpu_mark_segment_begins_int(arr, markers):
    i = cuda.grid(1)
    if i == 0:
        markers[0] = 1
    elif 0 < i < markers.size:
        if not markers[i]:
            markers[i] = not check_equals_int(arr[i], arr[i - 1])


@cuda.jit
def gpu_scatter_segment_begins(markers, scanned, begins):
    i = cuda.grid(1)
    if i < markers.size:
        if markers[i]:
            idx = scanned[i]
            begins[idx] = i


@cuda.jit
def gpu_mark_seg_segments(begins, markers):
    i = cuda.grid(1)
    if i < begins.size:
        markers[begins[i]] = 1


def find_segments(arr, segs=None, markers=None):
    """Find beginning indices of runs of equal values.

    Parameters
    ----------
    arr : device array
        The operand.
    segs : optional; device array
        Segment offsets that must exist in the output.

    Returns
    -------
    starting_indices : device array
        The starting indices of start of segments.
        Total segment count will be equal to the length of this.

    """
    # Compute diffs of consecutive elements
    null_markers = markers is None
    if null_markers:
        markers = zeros(arr.size, dtype=np.int32)
    else:
        assert markers.size == arr.size
        assert markers.dtype == np.dtype(np.int32), markers.dtype

    if markers.size > 0:
        if arr.dtype in ('float32', 'float64'):
            gpu_mark_segment_begins_float.forall(markers.size)(arr, markers)
        else:
            gpu_mark_segment_begins_int.forall(markers.size)(arr, markers)

    if segs is not None and null_markers and segs.size > 0:
        gpu_mark_seg_segments.forall(segs.size)(segs, markers)
    # Compute index of marked locations
    slots = prefixsum(markers)
    ct = slots[slots.size - 1]
    scanned = slots[:-1]
    # Compact segments
    begins = rmm.device_array(shape=int(ct), dtype=np.int32)
    if markers.size > 0:
        gpu_scatter_segment_begins.forall(markers.size)(markers, scanned,
                                                        begins)
    return begins, markers


@cuda.jit
def gpu_value_counts(arr, counts, total_size):
    i = cuda.grid(1)
    if 0 <= i < arr.size - 1:
        counts[i] = arr[i+1] - arr[i]
    elif i == arr.size - 1:
        counts[i] = total_size - arr[i]


def value_count(arr, total_size):
    counts = rmm.device_array(shape=len(arr), dtype=np.intp)
    if arr.size > 0:
        gpu_value_counts.forall(arr.size)(arr, counts, total_size)
    return counts


@cuda.jit
def gpu_recode(newdata, data, record_table, na_value):
    for i in range(cuda.threadIdx.x, data.size, cuda.blockDim.x):
        val = data[i]
        newval = (record_table[val]
                  if 0 <= val < record_table.size
                  else na_value)
        newdata[i] = newval


def recode(data, recode_table, na_value):
    """Recode data with the given recode table.
    And setting out-of-range values to *na_value*
    """
    newdata = rmm.device_array_like(data)
    recode_table = to_device(recode_table)
    blksz = 32 * 4
    blkct = min(16, max(1, data.size // blksz))
    gpu_recode[blkct, blksz](newdata, data, recode_table, na_value)
    return newdata


@cuda.jit
def gpu_row_matrix(rowmatrix, col, nrow, ncol):
    i = cuda.grid(1)
    if i < rowmatrix.size:
        rowmatrix[i] = col[i]


def row_matrix(cols, nrow, ncol, dtype):
    matrix = rmm.device_array(shape=(nrow, ncol), dtype=dtype, order='C')
    for colidx, col in enumerate(cols):
        data = matrix[:, colidx]
        if data.size > 0:
            gpu_row_matrix.forall(data.size)(data, col.to_gpu_array(), nrow,
                                             ncol)
    return matrix


@cuda.jit
def gpu_modulo(inp, out, d):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = inp[i] % d


def modulo(arr, d):
    """Array element modulo operator"""
    out = rmm.device_array(shape=arr.shape, dtype=arr.dtype)
    if arr.size > 0:
        gpu_modulo.forall(arr.size)(arr, out, d)
    return out


def boolean_array_to_index_array(bool_array):
    """ Converts a boolean array to an integer array to be used for gather /
        scatter operations
    """
    boolbits = compact_mask_bytes(bool_array)
    indices = arange(len(bool_array))
    _, selinds = copy_to_dense(indices, mask=boolbits)
    return selinds

# Copyright (c) 2018, NVIDIA CORPORATION.

from functools import lru_cache

import numpy as np
from numba import cuda, int32, numpy_support

import rmm

from cudf.utils.utils import (
    check_equals_float,
    check_equals_int,
    make_mask,
    mask_bitsize,
    mask_get,
    mask_set,
    rint,
)


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
        raise ValueError(
            msgfmt.format(
                size=size, start=start, stop=stop, step=step, dtype=dtype
            )
        )
    out = rmm.device_array(shape=int(size), dtype=dtype)
    if size > 0:
        gpu_arange.forall(size)(start, size, step, out)
    return out


@cuda.jit
def gpu_arange_reversed(size, out):
    i = cuda.grid(1)
    if i < size:
        out[i] = size - i - 1


def arange_reversed(size, dtype=np.int32):
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


def copy_array(arr, out=None):
    if out is None:
        out = rmm.device_array_like(arr)

    if (
        arr.is_c_contiguous()
        and out.is_c_contiguous()
        and out.size == arr.size
    ):
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
    import cudf._lib as libcudf

    from cudf.core.column import as_column

    # Allocate output
    slots = rmm.device_array(shape=vals.size + 1, dtype=vals.dtype)
    # Fill 0 to slot[0]
    gpu_fill_value[1, 1](slots[:1], 0)

    # Compute prefixsum on the mask
    in_col = as_column(vals)
    out_col = as_column(slots[1:])
    libcudf.reduce.scan(in_col, out_col, "sum", inclusive=True)
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
            raise ValueError("output array too small")
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


def make_empty_mask(size):
    bits = make_mask(size)
    if bits.size > 0:
        gpu_fill_value.forall(bits.size)(bits, 0)
    return bits


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
        out[i] = arr[i] == val


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
            out_col[i] = in_col[i - N]
        if i < N:
            out_col[i] = -1
    else:
        if i <= (in_col.size + N):
            out_col[i] = in_col[i - N]
        if i >= (in_col.size + N) and i < in_col.size:
            out_col[i] = -1


@cuda.jit
def gpu_diff(in_col, out_col, N):
    """Calculate the difference between values at positions i and i - N in an
    array and store the output in a new array.
    """
    i = cuda.grid(1)

    if N > 0:
        if i < in_col.size:
            out_col[i] = in_col[i] - in_col[i - N]
        if i < N:
            out_col[i] = -1
    else:
        if i <= (in_col.size + N):
            out_col[i] = in_col[i] - in_col[i - N]
        if i >= (in_col.size + N) and i < in_col.size:
            out_col[i] = -1


@cuda.jit
def gpu_round(in_col, out_col, decimal):
    i = cuda.grid(1)
    f = 10 ** decimal

    if i < in_col.size:
        ret = in_col[i] * f
        ret = rint(ret)
        tmp = ret / f
        out_col[i] = tmp


def apply_round(data, decimal):
    output_dary = rmm.device_array_like(data)
    if output_dary.size > 0:
        gpu_round.forall(output_dary.size)(data, output_dary, decimal)
    return output_dary


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
            raise NotImplementedError("k >= {}".format(MAX_FAST_UNIQUE_K))
        # setup mem
        outsz_ptr = rmm.device_array(shape=1, dtype=np.intp)
        out = rmm.device_array_like(arr)
        # kernel
        self._kernel[1, 64](arr, k, out, outsz_ptr)
        # copy to host
        unique_ct = outsz_ptr.copy_to_host()[0]
        if unique_ct < 0:
            raise ValueError("too many unique value (hint: increase k)")
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


@cuda.jit
def gpu_mark_found_int(arr, val, out, not_found):
    i = cuda.grid(1)
    if i < arr.size:
        if check_equals_int(arr[i], val):
            out[i] = i
        else:
            out[i] = not_found


@cuda.jit
def gpu_mark_found_float(arr, val, out, not_found):
    i = cuda.grid(1)
    if i < arr.size:
        if check_equals_float(arr[i], val):
            out[i] = i
        else:
            out[i] = not_found


@cuda.jit
def gpu_mark_gt(arr, val, out, not_found):
    i = cuda.grid(1)
    if i < arr.size:
        if arr[i] > val:
            out[i] = i
        else:
            out[i] = not_found


@cuda.jit
def gpu_mark_lt(arr, val, out, not_found):
    i = cuda.grid(1)
    if i < arr.size:
        if arr[i] < val:
            out[i] = i
        else:
            out[i] = not_found


def find_first(arr, val, compare="eq"):
    """
    Returns the index of the first occurrence of *val* in *arr*..
    Or the first occurence of *arr* *compare* *val*, if *compare* is not eq
    Otherwise, returns -1.

    Parameters
    ----------
    arr : device array
    val : scalar
    compare: str ('gt', 'lt', or 'eq' (default))
    """
    found = rmm.device_array_like(arr)
    if found.size > 0:
        if compare == "gt":
            gpu_mark_gt.forall(found.size)(arr, val, found, arr.size)
        elif compare == "lt":
            gpu_mark_lt.forall(found.size)(arr, val, found, arr.size)
        else:
            if arr.dtype in ("float32", "float64"):
                gpu_mark_found_float.forall(found.size)(
                    arr, val, found, arr.size
                )
            else:
                gpu_mark_found_int.forall(found.size)(
                    arr, val, found, arr.size
                )
    from cudf.core.column import as_column

    found_col = as_column(found)
    min_index = found_col.min()
    if min_index == arr.size:
        return -1
    else:
        return min_index


def find_last(arr, val, compare="eq"):
    """
    Returns the index of the last occurrence of *val* in *arr*.
    Or the last occurence of *arr* *compare* *val*, if *compare* is not eq
    Otherwise, returns -1.

    Parameters
    ----------
    arr : device array
    val : scalar
    compare: str ('gt', 'lt', or 'eq' (default))
    """
    found = rmm.device_array_like(arr)
    if found.size > 0:
        if compare == "gt":
            gpu_mark_gt.forall(found.size)(arr, val, found, -1)
        elif compare == "lt":
            gpu_mark_lt.forall(found.size)(arr, val, found, -1)
        else:
            if arr.dtype in ("float32", "float64"):
                gpu_mark_found_float.forall(found.size)(arr, val, found, -1)
            else:
                gpu_mark_found_int.forall(found.size)(arr, val, found, -1)
    from cudf.core.column import as_column

    found_col = as_column(found)
    max_index = found_col.max()
    return max_index


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
        if arr.dtype in ("float32", "float64"):
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
        gpu_scatter_segment_begins.forall(markers.size)(
            markers, scanned, begins
        )
    return begins, markers


@cuda.jit
def gpu_row_matrix(rowmatrix, col, nrow, ncol):
    i = cuda.grid(1)
    if i < rowmatrix.size:
        rowmatrix[i] = col[i]


def row_matrix(cols, nrow, ncol, dtype):
    matrix = rmm.device_array(shape=(nrow, ncol), dtype=dtype, order="C")
    for colidx, col in enumerate(cols):
        data = matrix[:, colidx]
        if data.size > 0:
            gpu_row_matrix.forall(data.size)(
                data, col.to_gpu_array(), nrow, ncol
            )
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


@cuda.jit
def gpu_window_sizes_from_offset(arr, window_sizes, offset):
    i = cuda.grid(1)
    j = i
    if i < arr.size:
        while j > -1:
            if (arr[i] - arr[j]) >= offset:
                break
            j -= 1
        window_sizes[i] = i - j


def window_sizes_from_offset(arr, offset):
    window_sizes = rmm.device_array(shape=(arr.shape), dtype="int32")
    if arr.size > 0:
        gpu_window_sizes_from_offset.forall(arr.size)(
            arr, window_sizes, offset
        )
    return window_sizes


@lru_cache(maxsize=32)
def compile_udf(udf, type_signature):
    """Copmile ``udf`` with `numba`

    Compile a python callable function ``udf`` with
    `numba.cuda.jit(device=True)` using ``type_signature`` into CUDA PTX
    together with the generated output type.

    The output is expected to be passed to the PTX parser in `libcudf`
    to generate a CUDA device funtion to be inlined into CUDA kernels,
    compiled at runtime and launched.

    Parameters
    --------
    udf:
      a python callable function

    type_signature:
      a tuple that specifies types of each of the input parameters of ``udf``.
      The types should be one in `numba.types` and could be converted from
      numpy types with `numba.numpy_support.from_dtype(...)`.

    Returns
    --------
    ptx_code:
      The compiled CUDA PTX

    output_type:
      An numpy type

    """
    decorated_udf = cuda.jit(udf, device=True)
    compiled = decorated_udf.compile(type_signature)
    ptx_code = decorated_udf.inspect_ptx(type_signature).decode("utf-8")
    output_type = numpy_support.as_dtype(compiled.signature.return_type)
    return (ptx_code, output_type.type)

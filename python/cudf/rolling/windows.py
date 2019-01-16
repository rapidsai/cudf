import numpy as np
import math
from numba import cuda
import cmath


@cuda.jit(device=True)
def window_kernel(shared, history_len, future_len, out_arr, window_size,
                  forward_window_size, arr_len, offset, min_size):
    """
    This function is to do window computation. Only one thread is assumed
    to do the computation. This thread is responsible for the `arr_len` of
    elements in the input array, which is cached in the `shared` array.
    Due to the limitation of numba, shared array cannot be sliced properly. As
    a work around, the passed-in `offset` position specify the beginning of
    the input array.

    Arguments:
        shared: a chunk of the array stored in shared memory. The first
                element of the data starts at `offset`. It has `history_len`
                of historical records for the first elements computation.
        history_len: the history length, which is mostly `window_size - 1`. For
                the early elements of the array, the history can be shorter for
                the beginning elements of the original array.
        future_len: the total future elements length from array_len to
                the first element that the current thread is responsible for.
        out_arr: the output array of size `arr_len`.
        window_size: the window size for the window function
        arr_len: the length of the output array
        offset: the starting position of the input shared array for the
                current thread
        min_size: at least there are min_size of non NA elements to compuate
    """
    pass


@cuda.jit(device=True)
def ewma_mean_window(shared, history_len, out_arr, window_size, span,
                     arr_len, offset, min_size):
    """
    This function is to compute the exponetially-weighted moving average for
    the window. See `window_kernel` for detailed arguments
    """
    s = 0.0
    v = 0.0
    v_current = 0.0
    first = False
    alpha = 2 / (span + 1)
    lam = 1
    total_weight = 0
    counter = 0
    weight_scale = 1.0
    weight_scale_current = 1.0
    average_size = 0
    for i in range(arr_len):
        if i + history_len < span - 1:
            out_arr[i] = np.nan
        else:
            if not first:
                # print(i, i + history_len + 1, window_size, history_len)
                for j in range(0, min(i + history_len + 1,
                                      window_size)):
                    if (cmath.isnan(shared[i + offset - j])) :
                        v = 0.0
                        weight_scale = 0.0
                    else:
                        v = shared[i + offset - j]
                        weight_scale = 1.0
                        average_size += 1
                    s += v * lam
                    counter += 1
                    total_weight += lam * weight_scale
                    lam *= (1 - alpha)
                if average_size >= min_size:
                    out_arr[i] = s / total_weight
                else:
                    out_arr[i] = np.nan
                first = True
            else:
                if (cmath.isnan(shared[i + offset])) :
                    v_current = 0.0
                    weight_scale_current = 0.0
                else:
                    v_current = shared[i + offset]
                    weight_scale_current = 1.0
                    average_size += 1
                if counter >= window_size:
                    if (cmath.isnan(shared[i + offset - window_size])) :
                        v = 0.0
                        weight_scale = 0.0
                    else:
                        v = shared[i + offset - window_size]
                        weight_scale = 1.0
                        average_size -= 1
                    s -= v * lam / (1 - alpha)
                    total_weight -= lam / (1 - alpha) * weight_scale
                else:
                    counter += 1
                    #total_weight += lam * weight_scale_current
                    lam *= (1 - alpha)
                total_weight *= (1 - alpha)
                total_weight += 1.0 * weight_scale_current
                s *= (1 - alpha)
                s += v_current
                if average_size >= min_size:
                    out_arr[i] = s / total_weight
                else:
                    out_arr[i] = np.nan


@cuda.jit(device=True)
def mean_window(shared, history_len, future_len, out_arr, window_size,
                  forward_window_size, arr_len, offset, min_size):
    """
    This function is to compute the moving average for the window
    See `window_kernel` for detailed arguments
    """
    first = False
    s = 0.0
    average_size = 0
    for i in range(arr_len):
        if i + history_len < window_size-1:
            out_arr[i] = np.nan
        elif future_len - i < forward_window_size + 1:
            out_arr[i] = np.nan
        else:
            if not first:
                for j in range(0, window_size + forward_window_size):
                    if not (cmath.isnan(
                        shared[offset + i - j + forward_window_size])):
                        s += shared[offset + i - j + forward_window_size]
                        average_size += 1
                if average_size >= min_size:
                    out_arr[i] = s / np.float64(average_size)
                else:
                    out_arr[i] = np.nan
                first = True
            else:
                if not (cmath.isnan(
                    shared[offset + i + forward_window_size])):
                    s += shared[offset + i + forward_window_size]
                    average_size += 1
                if not (cmath.isnan(
                    shared[offset + i - window_size])):
                    s -= shared[offset + i - window_size]
                    average_size -= 1
                if average_size >= min_size:
                    out_arr[i] = s / np.float64(average_size)
                else:
                    out_arr[i] = np.nan

@cuda.jit(device=True)
def var_window(shared, history_len, future_len, out_arr, window_size,
                  forward_window_size, arr_len, offset, min_size):
    """
    This function is to compute the var for the window
    See `window_kernel` for detailed arguments
    """
    s = 0.0 # this is mean
    var = 0.0 # this is variance
    first = False
    average_size = 0
    for i in range(arr_len):
        if i + history_len < window_size-1:
            out_arr[i] = np.nan
        elif future_len - i < forward_window_size + 1:
            out_arr[i] = np.nan
        else:
            if not first:
                for j in range(0, window_size + forward_window_size):
                    if not (cmath.isnan(
                        shared[offset + i - j + forward_window_size])):
                        s += shared[offset + i - j + forward_window_size]
                        var += shared[offset + i - j + forward_window_size] * shared[offset + i - j + forward_window_size]
                        average_size += 1
                if average_size >= min_size:
                    out_arr[i] = (var - s * s / np.float64(average_size) ) / np.float64(average_size - 1.0)
                else:
                    out_arr[i] = np.nan
                first = True
            else:
                if not (cmath.isnan(
                    shared[offset + i + forward_window_size])):
                    s += shared[offset + i + forward_window_size]
                    var += shared[offset + i + forward_window_size] * shared[offset + i + forward_window_size]
                    average_size += 1
                if not (cmath.isnan(
                    shared[offset + i - window_size])):
                    s -= shared[offset + i - window_size]
                    var -= shared[offset + i - window_size] * shared[offset + i - window_size]
                    average_size -= 1
                if average_size >= min_size:
                    out_arr[i] = (var - s * s / np.float64(average_size) ) / np.float64(average_size - 1.0)
                else:
                    out_arr[i] = np.nan

@cuda.jit(device=True)
def std_window(shared, history_len, future_len, out_arr, window_size,
                  forward_window_size, arr_len, offset, min_size):
    """
    This function is to compute the std for the window
    See `window_kernel` for detailed arguments
    """
    s = 0.0 # this is mean
    var = 0.0 # this is variance
    first = False
    average_size = 0
    for i in range(arr_len):
        if i + history_len < window_size-1:
            out_arr[i] = np.nan
        elif future_len - i < forward_window_size + 1:
            out_arr[i] = np.nan
        else:
            if not first:
                for j in range(0, window_size + forward_window_size):
                    if not (cmath.isnan(
                        shared[offset + i - j + forward_window_size])):
                        s += shared[offset + i - j + forward_window_size]
                        var += shared[offset + i - j + forward_window_size] * shared[offset + i - j + forward_window_size]
                        average_size += 1
                if average_size >= min_size:
                    v = math.sqrt(abs((var - s * s / np.float64(average_size) ) / np.float64(average_size - 1.0)))
                    out_arr[i] = v
                else:
                    out_arr[i] = np.nan
                first = True
            else:
                if not (cmath.isnan(
                    shared[offset + i + forward_window_size])):
                    s += shared[offset + i + forward_window_size]
                    var += shared[offset + i + forward_window_size] * shared[offset + i + forward_window_size]
                    average_size += 1
                if not (cmath.isnan(
                    shared[offset + i - window_size])):
                    s -= shared[offset + i - window_size]
                    var -= shared[offset + i - window_size] * shared[offset + i - window_size]
                    average_size -= 1
                if average_size >= min_size:
                    v = math.sqrt(abs((var - s * s / np.float64(average_size) ) / np.float64(average_size - 1.0)))
                    out_arr[i] = v
                else:
                    out_arr[i] = np.nan



@cuda.jit(device=True)
def sum_window(shared, history_len, future_len, out_arr, window_size,
                  forward_window_size, arr_len, offset, min_size):
    """
    This function is to compute the sum for the window
    See `window_kernel` for detailed arguments
    """
    first = False
    s = 0.0
    average_size = 0
    for i in range(arr_len):
        if i + history_len < window_size-1:
            out_arr[i] = np.nan
        elif future_len - i < forward_window_size + 1:
            out_arr[i] = np.nan
        else:
            if not first:
                for j in range(0, window_size + forward_window_size):
                    if not (cmath.isnan(
                        shared[offset + i - j + forward_window_size])):
                        s += shared[offset + i - j + forward_window_size]
                        average_size += 1
                if average_size >= min_size:
                    out_arr[i] = s
                else:
                    out_arr[i] = np.nan
                first = True
            else:
                if not (cmath.isnan(
                    shared[offset + i + forward_window_size])):
                    s += shared[offset + i + forward_window_size]
                    average_size += 1
                if not (cmath.isnan(
                    shared[offset + i - window_size])):
                    s -= shared[offset + i - window_size]
                    average_size -= 1
                if average_size >= min_size:
                    out_arr[i] = s
                else:
                    out_arr[i] = np.nan


@cuda.jit(device=True)
def max_window(shared, history_len, future_len, out_arr, window_size,
                  forward_window_size, arr_len, offset, min_size):
    """
    This function is to compute the max for the window
    See `window_kernel` for detailed arguments
    """
    for i in range(arr_len):
        if i + history_len < window_size-1:
            out_arr[i] = np.nan
        elif future_len - i < forward_window_size + 1:
            out_arr[i] = np.nan
        else:
            s = -np.inf  # maximum
            average_size = 0
            for j in range(0, window_size + forward_window_size):
                if not (cmath.isnan(
                    shared[offset + i - j + forward_window_size])):
                    # bigger than the max
                    if shared[i + offset - j + forward_window_size] > s:
                        s = shared[i + offset - j + forward_window_size]
                    average_size += 1
            if average_size >= min_size:
                out_arr[i] = s
            else:
                out_arr[i] = np.nan


@cuda.jit(device=True)
def min_window(shared, history_len, future_len, out_arr, window_size,
                  forward_window_size, arr_len, offset, min_size):
    """
    This function is to compute the min for the window
    See `window_kernel` for detailed arguments
    """
    for i in range(arr_len):
        if i + history_len < window_size-1:
            out_arr[i] = np.nan
        elif future_len - i < forward_window_size + 1:
            out_arr[i] = np.nan
        else:
            s = np.inf  # minimum
            average_size = 0
            for j in range(0, window_size + forward_window_size):
                if not (cmath.isnan(
                    shared[offset + i - j + forward_window_size])):
                    # smaller than the min
                    if shared[i + offset - j + forward_window_size] < s:
                        s = shared[i + offset - j + forward_window_size]
                    average_size += 1
            if average_size >= min_size:
                out_arr[i] = s
            else:
                out_arr[i] = np.nan


@cuda.jit(device=True)
def backward_diff_window(shared, history_len, future_len, out_arr, window_size,
                  forward_window_size, arr_len, offset, min_size):
    """
    This function is to compute the backward element difference.
    See `window_kernel` for detailed arguments
    """
    for i in range(arr_len):
        if i + history_len < window_size-1:
            out_arr[i] = np.nan
        elif future_len - i < forward_window_size + 1:
            out_arr[i] = np.nan
        else:
            if (cmath.isnan(shared[offset + i]) or cmath.isnan(shared[offset + i - window_size + 1])):
                out_arr[i] = np.nan
            else:
                out_arr[i] = shared[offset + i] - shared[offset + i
                                                         - window_size + 1]


@cuda.jit(device=True)
def backward_shift_window(shared, history_len, future_len, out_arr, window_size,
                          forward_window_size, arr_len, offset, min_size):
    """
    This function is to shfit elements backward
    See `window_kernel` for detailed arguments
    """
    for i in range(arr_len):
        if i + history_len < window_size-1:
            out_arr[i] = np.nan
        elif future_len - i < forward_window_size + 1:
            out_arr[i] = np.nan
        else:
            if (cmath.isnan(
                shared[offset + i - window_size + 1])):
                out_arr[i] = np.nan
            else:
                out_arr[i] = shared[offset + i - window_size + 1]


@cuda.jit(device=True)
def forward_diff_window(shared, history_len, future_len, out_arr, window_size,
                        forward_window_size, arr_len, offset, min_size):
    """
    This function is to compute the forward element difference.
    See `window_kernel` for detailed arguments
    """
    for i in range(arr_len):
        if i + history_len < window_size-1:
            out_arr[i] = np.nan
        elif future_len - i < forward_window_size + 1:
            out_arr[i] = np.nan
        else:
            if (cmath.isnan(shared[offset + i]) or cmath.isnan(shared[offset + i + forward_window_size])):
                out_arr[i] = np.nan
            else:
                out_arr[i] = shared[offset + i] - shared[offset + i + forward_window_size]


@cuda.jit(device=True)
def forward_shift_window(shared, history_len, future_len, out_arr, window_size,
                         forward_window_size, arr_len, offset, min_size):
    """
    This function is to compute the forward element difference.
    See `window_kernel` for detailed arguments
    """
    for i in range(arr_len):
        if i + history_len < window_size-1:
            out_arr[i] = np.nan
        elif future_len - i < forward_window_size + 1:
            out_arr[i] = np.nan
        else:
            if (cmath.isnan(
                shared[offset + i + forward_window_size])):
                out_arr[i] = np.nan
            else:
                out_arr[i] = shared[offset + i + forward_window_size]

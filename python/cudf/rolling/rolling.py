from numba import cuda
import numba
from cudf.rolling.windows import (mean_window, std_window, var_window,
                                  max_window,
                                  min_window, sum_window, backward_diff_window,
                                  backward_shift_window, forward_diff_window,
                                  forward_shift_window)


kernel_cache = {}


def get_rolling_kernel(method):

    if method in kernel_cache:
        return kernel_cache[method]

    @cuda.jit
    def kernel(in_arr, out_arr, backward_length, forward_length,
               arr_len, thread_tile, min_size):
        """
        This kernel is to copy input array elements into shared array.
        The total window size is backward_length + forward_length. To compute
        output element at i, it uses [i - backward_length - 1, i] elements in
        history, and [i + 1, i + forward_lengh] elements in the future.
        Arguments:
            in_arr: input gpu array
            out_arr: output gpu_array
            backward_length: the history elements in the windonw
            forward_length: the forward elements in the window
            arr_len: the input/output array length
            thread_tile: each thread is responsible for `thread_tile` number
                         of elements
            min_size: the minimum number of non-na elements
        """
        shared = cuda.shared.array(shape=0,
                                   dtype=numba.float64)
        block_size = cuda.blockDim.x  # total number of threads
        tx = cuda.threadIdx.x
        # Block id in a 1D grid
        bid = cuda.blockIdx.x
        starting_id = bid * block_size * thread_tile

        # copy the thread_tile * number_of_thread_per_block into the shared
        for j in range(thread_tile):
            offset = tx + j * block_size
            if (starting_id + offset) < arr_len:
                shared[offset + backward_length - 1] = in_arr[
                    starting_id + offset]
            cuda.syncthreads()

        # copy the backward_length - 1 into the shared
        for j in range(0, backward_length - 1, block_size):
            if (((tx + j) <
                 backward_length - 1) and (
                     starting_id - backward_length + 1 + tx + j >= 0)):
                shared[tx + j] = in_arr[starting_id
                                        - backward_length + 1 + tx + j]
            cuda.syncthreads()
        # copy the forward_length into the shared
        for j in range(0, forward_length, block_size):
            element_id = (starting_id + thread_tile * block_size + tx + j)
            if (((tx + j) < forward_length) and (element_id < arr_len)):
                shared[thread_tile * block_size + backward_length - 1 + tx +
                       j] = in_arr[element_id]
            cuda.syncthreads()
        # slice the shared memory for each threads
        start_shared = tx * thread_tile
        his_len = min(backward_length - 1,
                      starting_id + tx * thread_tile)
        future_len = max(arr_len - (starting_id + tx * thread_tile), 0)

        # slice the global memory for each threads
        start = starting_id + tx * thread_tile
        end = min(starting_id + (tx + 1) * thread_tile, arr_len)
        sub_outarr = out_arr[start:end]
        sub_len = end - start
        method(shared, his_len, future_len, sub_outarr,
               backward_length, forward_length,
               sub_len, backward_length - 1 + start_shared,
               min_size
               )
    kernel_cache[method] = kernel
    return kernel


class Rolling(object):

    def __init__(self, window, input_arr, min_periods=None, forward_window=0,
                 thread_tile=48, number_of_threads=64):
        """
        The Rolling class that is used to do rolling window computations.
        The window size is `window + forward_window`. The element i uses
        [i - window -1, i + forward_window] elements to do the window
        computation.
        Arguments:
            window: the history window size.
            input_arr: the input GPU array or cudf.Series
            min_periods: the minimum number of non-na elements need to get an
                         output
            forward_window: the windows size in the forward direction
            thread_tile: each thread will be responsible for `thread_tile`
                         number of elements in window computation
            number_of_threads: number of threads in a block for CUDA
                               computation
        """
        if isinstance(input_arr, numba.cuda.cudadrv.devicearray.DeviceNDArray):
            self.gpu_in = input_arr
        else:
            self.gpu_in = input_arr.to_gpu_array()
        if min_periods is None:
            self.min_periods = window + forward_window
        else:
            self.min_periods = min_periods
        self.window = window
        self.forward_window = forward_window
        self.number_of_threads = number_of_threads
        self.array_len = len(self.gpu_in)
        self.gpu_out = numba.cuda.device_array_like(self.gpu_in)
        self.thread_tile = thread_tile
        self.number_of_blocks = (self.array_len +
                                 (number_of_threads * thread_tile - 1)) // (
                                     number_of_threads * thread_tile)

        self.shared_buffer_size = (self.number_of_threads * self.thread_tile
                                   + self.window - 1 + self.forward_window)

    def apply(self, method):
        gpu_out = numba.cuda.device_array_like(self.gpu_in)
        kernel = get_rolling_kernel(method)
        kernel[(self.number_of_blocks,),
               (self.number_of_threads,),
               0,
               self.shared_buffer_size * 8](self.gpu_in,
                                            gpu_out,
                                            self.window,
                                            self.forward_window,
                                            self.array_len,
                                            self.thread_tile,
                                            self.min_periods)
        return gpu_out

    def mean(self):
        return self.apply(mean_window)

    def std(self):
        return self.apply(std_window)

    def var(self):
        return self.apply(var_window)

    def max(self):
        return self.apply(max_window)

    def min(self):
        return self.apply(min_window)

    def sum(self):
        return self.apply(sum_window)

    def backward_diff(self):
        return self.apply(backward_diff_window)

    def backward_shift(self):
        return self.apply(backward_shift_window)

    def forward_diff(self):
        return self.apply(forward_diff_window)

    def forward_shift(self):
        return self.apply(forward_shift_window)

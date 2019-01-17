from numba import cuda
import numba
from cudf.rolling.windows import (ewma_mean_window)


kernel_cache = {}


def get_ewm_kernel(method):

    if method in kernel_cache:
        return kernel_cache[method]

    @cuda.jit
    def kernel(in_arr, out_arr, average_length, span, arr_len, thread_tile,
               min_size):
        """
        This kernel is to copy input array elements into shared array.
        The total window size. To compute
        output element at i, it uses [i - average_length - 1, i] elements in
        history.
        Arguments:
            in_arr: input gpu array
            out_arr: output gpu_array
            average_length: is the size used to compute expoential weighted
                            average
            span: the span size for the exponential weighted average
            arr_len: the input/output array length
            thread_tile: each thread is responsible for `thread_tile` number
                         of elements
            min_size: the minimum number of non-na elements
        """
        shared = cuda.shared.array(shape=0,
                                   dtype=numba.float64)
        block_size = cuda.blockDim.x
        tx = cuda.threadIdx.x
        # Block id in a 1D grid
        bid = cuda.blockIdx.x
        starting_id = bid * block_size * thread_tile

        # copy the thread_tile * number_of_thread_per_block into the shared
        for j in range(thread_tile):
            offset = tx + j * block_size
            if (starting_id + offset) < arr_len:
                shared[offset + average_length - 1] = in_arr[
                    starting_id + offset]
            cuda.syncthreads()
        # copy the average_length - 1 into the shared
        for j in range(0, average_length - 1, block_size):
            if (((tx + j) <
                 average_length - 1) and (
                     starting_id - average_length + 1 + tx + j >= 0)):
                shared[tx + j] = in_arr[starting_id
                                        - average_length + 1 + tx + j]
        cuda.syncthreads()
        # slice the shared memory for each threads
        start_shared = tx * thread_tile
        his_len = min(average_length - 1,
                      starting_id + tx * thread_tile)

        # slice the global memory for each threads
        start = starting_id + tx * thread_tile
        end = min(starting_id + (tx + 1) * thread_tile, arr_len)
        sub_outarr = out_arr[start:end]
        sub_len = end - start
        method(shared, his_len, sub_outarr,
               average_length, span, sub_len,
               average_length - 1 + start_shared,
               min_size)
    kernel_cache[method] = kernel
    return kernel


class Ewm(object):

    def __init__(self, span, input_arr, min_periods=None, thread_tile=48,
                 number_of_threads=64, expand_multiplier=10):
        """
        The Ewm class that is used to do rolling exponential weighted moving
        average. It uses expand_multiplier * span elements to do the weighted
        average. So adjust expand_multiplier to adjust accuracy.
        Arguments:
            span: the span parameter in the exponential weighted moving average
            input_arr: the input GPU array or cudf.Series
            min_periods: the minimum number of non-na elements need to get an
                         output
            thread_tile: each thread will be responsible for `thread_tile`
                         number of elements in window computation
            number_of_threads: number of threads in a block for CUDA
                               computation
            expand_multiplier: the number of elements used computing EWM is
                                controled by this constant. The higher this
                                number, the better the accuracy but slower in
                                performance
        """
        if isinstance(input_arr, numba.cuda.cudadrv.devicearray.DeviceNDArray):
            self.gpu_in = input_arr
        else:
            self.gpu_in = input_arr.to_gpu_array()
        if min_periods is None:
            self.min_periods = span
        else:
            self.min_periods = min_periods
        self.span = span
        self.window = span * expand_multiplier
        self.number_of_threads = number_of_threads
        self.array_len = len(self.gpu_in)
        self.thread_tile = thread_tile
        self.number_of_blocks = (self.array_len +
                                 (number_of_threads * thread_tile - 1)) // (
                                     number_of_threads * thread_tile)

        self.shared_buffer_size = (self.number_of_threads * self.thread_tile
                                   + self.window - 1)

    def apply(self, method):
        gpu_out = numba.cuda.device_array_like(self.gpu_in)
        kernel = get_ewm_kernel(method)
        kernel[(self.number_of_blocks,),
               (self.number_of_threads,),
               0,
               self.shared_buffer_size * 8](self.gpu_in,
                                            gpu_out,
                                            self.window,
                                            self.span,
                                            self.array_len,
                                            self.thread_tile,
                                            self.min_periods)
        return gpu_out

    def mean(self):
        return self.apply(ewma_mean_window)

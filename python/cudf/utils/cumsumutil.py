from numba import cuda
import numba


@cuda.jit
def cumsum_kernel(in_arr, out_arr, block_arr, arr_len):
    """
    implemented the algorithm at
    https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
    tested the bank non conflict version, no performance difference
    """
    shared = cuda.shared.array(shape=0, dtype=numba.float64)
    num_threads = cuda.blockDim.x
    tx = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    partial_sum_offset = num_threads * 2
    starting_id = bid * partial_sum_offset

    # load the in_arr to shared
    for j in range(2):
        offset = tx + j * num_threads
        if (offset + starting_id) < arr_len:
            shared[offset] = in_arr[offset + starting_id]
        else:
            shared[offset] = 0.0
        cuda.syncthreads()

    offset = 1

    d = num_threads

    while d > 0:
        cuda.syncthreads()
        if (tx < d):
            ai = offset*(2*tx+1)-1
            bi = offset*(2*tx+2)-1
            shared[bi] += shared[ai]
        offset *= 2
        d = d // 2

    if (tx == 0):
        block_arr[bid] = shared[2 * num_threads - 1]
        shared[2 * num_threads - 1] = 0.0

    d = 1
    while d < 2 * num_threads:
        offset = offset // 2
        cuda.syncthreads()
        if tx < d:
            ai = offset*(2*tx+1)-1
            bi = offset*(2*tx+2)-1
            t = shared[ai]
            shared[ai] = shared[bi]
            shared[bi] += t
        d *= 2
    cuda.syncthreads()

    # load back to the output
    for j in range(2):
        offset = tx + j * num_threads
        if (offset + starting_id) < arr_len and offset + 1 < 2 * num_threads:
            out_arr[offset + starting_id] = shared[offset + 1]

    if tx == 0:
        arr_id = min(arr_len - 1, starting_id + 2 * num_threads - 1)
        out_arr[arr_id] = block_arr[bid]


@cuda.jit
def correct_kernel(in_arr, block_arr, arr_len):
    num_threads = cuda.blockDim.x
    tx = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    partial_sum_offset = num_threads * 2
    starting_id = bid * partial_sum_offset

    for j in range(2):
        offset = tx + j * num_threads
        lookup = bid - 1

        if lookup >= 0 and (offset + starting_id) < arr_len:
            in_arr[offset + starting_id] += block_arr[lookup]


def cumsum(input_arr, number_of_threads=1024):
    """
    compute the cumsum of the numba gpu array
    parameters:
        input_arr: array of Numba GPU array type
    returns:
        cumsum output in Numba GPU array
    """
    array_len = len(input_arr)
    number_of_blocks = (array_len + (
        number_of_threads * 2 - 1)) // (number_of_threads * 2)

    shared_buffer_size = (number_of_threads * 2)

    block_summary = numba.cuda.device_array(number_of_blocks)
    gpu_out = numba.cuda.device_array_like(input_arr)
    cumsum_kernel[(number_of_blocks,),
                  (number_of_threads,),
                  0,
                  shared_buffer_size * 8](input_arr,
                                          gpu_out,
                                          block_summary,
                                          array_len)
    if (number_of_blocks == 1):
        return gpu_out
    else:
        block_sum = cumsum(block_summary)
        correct_kernel[(number_of_blocks,),
                       (number_of_threads,)](gpu_out,
                                             block_sum,
                                             array_len)
        return gpu_out

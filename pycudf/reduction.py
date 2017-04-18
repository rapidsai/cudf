from numba import cuda
from numba.numpy_support import from_dtype


def gpu_reduce_factory(fn, nbtype):
    gpu_functor = cuda.jit(device=True)(fn)
    warpsize = 32
    inner_sm_size = warpsize + 1   # plus one to avoid SM collision
    numwarps = 4

    def gpu_reduce_block_strided(arr, partials, init, numblocks):
        """
        Perform reductions on *arr* and writing out partial reduction result
        into *partials*.  The length of *partials* is determined by the
        number of threadblocks, which must be passed as *numblocks*.
        The initial value is set with *init*.

        Launch config:

        Blocksize must be mutiple of warpsize and it is limited to 4 warps.
        """
        tid = cuda.threadIdx.x
        blkid = cuda.blockIdx.x
        blksz = cuda.blockDim.x
        warpid = tid // warpsize
        laneid = tid % warpsize
        assert warpid < numwarps

        sm_partials = cuda.shared.array((numwarps, inner_sm_size), dtype=nbtype)

        # block strided loop to compute the reduction
        tmp = init
        for i in range(tid + blksz * blkid, arr.size, blksz * numblocks):
            # load
            val = arr[i]
            # compute
            tmp = gpu_functor(tmp, val)

        # store partial reduction for block reduction
        sm_partials[warpid, laneid] = tmp
        cuda.syncthreads()

        # inner-warp reduction
        sm_this = sm_partials[warpid, :]
        width = warpsize // 2
        while width:
            if laneid < width:
                old = sm_this[laneid]
                sm_this[laneid] = gpu_functor(old, sm_this[laneid + width])
            width //= 2

        cuda.syncthreads()
        # at this point, only the first slot for each warp in tsm_partials
        # is valid.

        # finish up block reduction
        # warning: this is assuming 4 warps.
        assert numwarps == 4
        if tid < 2:
            sm_partials[tid, 0] = gpu_functor(sm_partials[tid, 0],
                                              sm_partials[tid + 2, 0])
        if tid == 0:
            partials[blkid] = gpu_functor(sm_partials[0, 0], sm_partials[1, 0])

    return cuda.jit(gpu_reduce_block_strided)


class Reduce(object):
    _cache = {}

    def __init__(self, functor):
        self._functor = functor

    def __call__(self, arr, init):
        key = self._functor, arr.dtype
        if key in self._cache:
            kernel = self._cache[key]
        else:
            kernel = gpu_reduce_factory(self._functor, from_dtype(arr.dtype))
            self._cache[key] = kernel

        blocksize = 4 * 32
        blockcount = 4
        partials = cuda.device_array(shape=blockcount, dtype=arr.dtype)
        # compute partials (1 per block)
        kernel[blockcount, blocksize](arr, partials, init, blockcount)
        # finish up
        kernel[1, blocksize](partials, partials, init, 1)
        return partials[0]

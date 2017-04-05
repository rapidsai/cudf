import numpy as np


mask_dtype = np.dtype(np.uint32)
mask_bitsize = mask_dtype.itemsize * 8


def calc_chunk_size(size, chunksize):
    return (size + chunksize - 1) // chunksize


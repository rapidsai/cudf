from collections import namedtuple

import numpy as np


mask_dtype = np.dtype(np.uint8)
mask_bitsize = mask_dtype.itemsize * 8


def calc_chunk_size(size, chunksize):
    return (size + chunksize - 1) // chunksize


_TypeMinMax = namedtuple('_TypeMinMax', 'min,max')


def get_numeric_type_info(dtype):
    if dtype.kind in 'iu':
        info = np.iinfo(dtype)
        return _TypeMinMax(info.min, info.max)
    elif dtype.kind in 'f':
        return _TypeMinMax(dtype.type('-inf'), dtype.type('+inf'))
    else:
        raise TypeError(dtype)


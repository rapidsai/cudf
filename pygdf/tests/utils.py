
import numpy as np
from pygdf import utils


def random_bitmask(size):
    """
    Parameters
    ----------
    size : int
        number of bits
    """
    sz = utils.calc_chunk_size(size, utils.mask_bitsize)
    data = np.random.randint(0, 255 + 1, size=sz)
    return data.astype(utils.mask_dtype)


def expand_bits_to_bytes(arr):
    def fix_binary(bstr):
        bstr = bstr[2:]
        diff = 8 - len(bstr)
        return ('0' * diff + bstr)[::-1]

    ba = bytearray(arr.data)
    return list(map(int, ''.join(map(fix_binary, map(bin, ba)))))


def count_zero(arr):
    arr = np.asarray(arr)
    return np.count_nonzero(arr == 0)

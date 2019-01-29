
import numpy as np
import pandas as pd
from cudf.utils import utils

import pandas.util.testing as tm


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


def assert_eq(a, b, **kwargs):
    """ Assert that two cudf-like things are equivalent

    This equality test works for pandas/cudf dataframes/series/indexes/scalars
    in the same way, and so makes it easier to perform parametrized testing
    without switching between assert_frame_equal/assert_series_equal/...
    functions.
    """
    if hasattr(a, 'to_pandas'):
        a = a.to_pandas()
    if hasattr(b, 'to_pandas'):
        b = b.to_pandas()
    if isinstance(a, pd.DataFrame):
        tm.assert_frame_equal(a, b, **kwargs)
    elif isinstance(a, pd.Series):
        tm.assert_series_equal(a, b, **kwargs)
    elif isinstance(a, pd.Index):
        tm.assert_index_equal(a, b, **kwargs)
    else:
        if a == b:
            return True
        else:
            if np.isnan(a):
                assert np.isnan(b)
            else:
                assert np.allclose(a, b)
    return True

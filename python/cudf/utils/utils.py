from collections import namedtuple

import numpy as np
import pandas as pd
import pyarrow as pa
from math import isnan, isinf, ceil

from numba import njit

from librmm_cffi import librmm as rmm


mask_dtype = np.dtype(np.int8)
mask_bitsize = mask_dtype.itemsize * 8
mask_byte_padding = 64


def calc_chunk_size(size, chunksize):
    return mask_byte_padding * \
           ceil(((size + chunksize - 1) // chunksize) / mask_byte_padding)


_TypeMinMax = namedtuple('_TypeMinMax', 'min,max')


def get_numeric_type_info(dtype):
    if dtype.kind in 'iu':
        info = np.iinfo(dtype)
        return _TypeMinMax(info.min, info.max)
    elif dtype.kind in 'f':
        return _TypeMinMax(dtype.type('-inf'), dtype.type('+inf'))
    else:
        raise TypeError(dtype)


@njit
def mask_get(mask, pos):
    return (mask[pos // mask_bitsize] >> (pos % mask_bitsize)) & 1


@njit
def mask_set(mask, pos):
    mask[pos // mask_bitsize] |= 1 << (pos % mask_bitsize)


@njit
def check_equals_float(a, b):
    return (a == b or (isnan(a) and isnan(b)) or
            ((isinf(a) and a < 0) and (isinf(b) and b < 0)) or
            ((isinf(a) and a > 0) and (isinf(b) and b > 0)))


@njit
def check_equals_int(a, b):
    return (a == b)


def make_mask(size):
    """Create mask to obtain at least *size* number of bits.
    """
    size = calc_chunk_size(size, mask_bitsize)
    return rmm.device_array(shape=size, dtype=mask_dtype)


def require_writeable_array(arr):
    # This should be fixed in numba (numba issue #2521)
    return np.require(arr, requirements='W')


def scalar_broadcast_to(scalar, shape, dtype):
    from cudf.utils.cudautils import fill_value

    if not isinstance(shape, tuple):
        shape = (shape,)

    if np.dtype(dtype) == np.dtype("object"):
        import nvstrings
        from cudf.dataframe.string import StringColumn
        from cudf.utils.cudautils import zeros
        gather_map = zeros(shape[0], dtype='int32')
        scalar_str_col = StringColumn(nvstrings.to_device([scalar]))
        return scalar_str_col[gather_map]
    else:
        da = rmm.device_array(shape, dtype=dtype)
        if da.size != 0:
            fill_value(da, scalar)
        return da


def normalize_index(index, size, doraise=True):
    """Normalize negative index
    """
    if index < 0:
        index = size + index
    if doraise and not (0 <= index < size):
        raise IndexError('out-of-bound')
    return min(index, size)


# borrowed from a wonderful blog:
# https://avilpage.com/2015/03/a-slice-of-python-intelligence-behind.html
def standard_python_slice(len_idx, arg):
    """ Figuring out the missing parameters of slice"""

    start = arg.start
    stop = arg.stop
    step = arg.step

    if step is None:
        step = 1
    if step == 0:
        raise Exception("Step cannot be zero.")

    if start is None:
        start = 0 if step > 0 else len_idx - 1
    else:
        if start < 0:
            start += len_idx
        if start < 0:
            start = 0 if step > 0 else -1
        if start >= len_idx:
            start = len_idx if step > 0 else len_idx - 1

    if stop is None:
        stop = len_idx if step > 0 else -1
    else:
        if stop < 0:
            stop += len_idx
        if stop < 0:
            stop = 0 if step > 0 else -1
        if stop >= len_idx:
            stop = len_idx if step > 0 else len_idx - 1

    if (step < 0 and stop >= start) or (step > 0 and start >= stop):
        slice_length = 0
    elif step < 0:
        slice_length = (stop - start + 1)//step + 1
    else:
        slice_length = (stop - start - 1)//step + 1

    return start, stop, step, slice_length


list_types_tuple = (list, np.array)


def buffers_from_pyarrow(pa_arr, dtype=None):
    from cudf.dataframe.buffer import Buffer
    from cudf.utils.cudautils import copy_array

    buffers = pa_arr.buffers()

    if buffers[0]:
        mask_dev_array = make_mask(len(pa_arr))
        arrow_dev_array = rmm.to_device(np.array(buffers[0]).view('int8'))
        copy_array(arrow_dev_array, mask_dev_array)
        pamask = Buffer(
            mask_dev_array
        )
    else:
        pamask = None

    if dtype:
        new_dtype = dtype
    else:
        if isinstance(pa_arr, pa.DictionaryArray):
            new_dtype = pa_arr.indices.type.to_pandas_dtype()
        else:
            new_dtype = pa_arr.type.to_pandas_dtype()

    if buffers[1]:
        padata = Buffer(
            np.array(buffers[1]).view(new_dtype)[
                pa_arr.offset:pa_arr.offset + len(pa_arr)
            ]
        )
    else:
        padata = Buffer(
            np.empty(0, dtype=new_dtype)
        )
    return (pamask, padata)


def cudf_dtype_from_pydata_dtype(dtype):
    """ Given a numpy or pandas dtype, converts it into the equivalent cuDF
        Python dtype.
    """
    try:
        # pd 0.24.X
        from pandas.core.dtypes.common import \
            infer_dtype_from_object
    except ImportError:
        # pd 0.23.X
        from pandas.core.dtypes.common import \
            _get_dtype_from_object as infer_dtype_from_object

    if pd.api.types.is_categorical_dtype(dtype):
        pass
    elif np.issubdtype(dtype, np.datetime64):
        dtype = np.datetime64

    return infer_dtype_from_object(dtype)


def get_dummies(df, prefix, prefix_sep='_', cats={}, columns=None,
                dtype='float64'):
    """ Returns a dataframe whose columns are the one hot encodings of all
    columns in `df`

    Parameters
    ----------
    df : cudf.DataFrame
        dataframe to encode
    prefix : str or dict
        prefix to append. Either a str (to apply a constant prefix) or a
        mapping of column names to prefixes
    prefix_sep : str, optional
        separator to use when appending prefixes
    cats : dict, optional
        dictionary mapping column names to sequences of integers representing
        that column's category. See `cudf.DataFrame.one_hot_encoding` for more
        information. if not supplied, it will be computed
    columns : sequence of str, optional
        Names of columns to encode. If not provided, will attempt to encode all
        columns. Note this is different from pandas default behavior, which
        encodes all columns with dtype object or categorical
    dtype : str, optional
        output dtyle, default 'float64'
    """
    from cudf.multi import concat

    if isinstance(prefix, str):
        prefix_map = {}
    else:
        prefix_map = prefix

    if columns is None:
        columns = df.columns

    return concat([
        df.one_hot_encoding(
            name,
            prefix=prefix_map.get(name, prefix),
            cats=cats.get(name, df[name].unique()),
            prefix_sep=prefix_sep,
            dtype=dtype)
        for name in columns
    ], axis=1)

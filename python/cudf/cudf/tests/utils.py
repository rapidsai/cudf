from contextlib import contextmanager

import cupy
import numpy as np
import pandas as pd
from pandas.util import testing as tm

import cudf
from cudf._lib.null_mask import bitmask_allocation_size_bytes
from cudf.utils import dtypes as dtypeutils

supported_numpy_dtypes = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "datetime64[ms]",
    "datetime64[us]",
]

SIGNED_INTEGER_TYPES = sorted(list(dtypeutils.SIGNED_INTEGER_TYPES))
UNSIGNED_TYPES = sorted(list(dtypeutils.UNSIGNED_TYPES))
INTEGER_TYPES = sorted(list(dtypeutils.INTEGER_TYPES))
FLOAT_TYPES = sorted(list(dtypeutils.FLOAT_TYPES))
SIGNED_TYPES = sorted(list(dtypeutils.SIGNED_TYPES))
NUMERIC_TYPES = sorted(list(dtypeutils.NUMERIC_TYPES))
DATETIME_TYPES = sorted(list(dtypeutils.DATETIME_TYPES))
OTHER_TYPES = sorted(list(dtypeutils.OTHER_TYPES))
ALL_TYPES = sorted(list(dtypeutils.ALL_TYPES))


def random_bitmask(size):
    """
    Parameters
    ----------
    size : int
        number of bits
    """
    sz = bitmask_allocation_size_bytes(size)
    data = np.random.randint(0, 255, dtype="u1", size=sz)
    return data.view("i1")


def expand_bits_to_bytes(arr):
    def fix_binary(bstr):
        bstr = bstr[2:]
        diff = 8 - len(bstr)
        return ("0" * diff + bstr)[::-1]

    ba = bytearray(arr.data)
    return list(map(int, "".join(map(fix_binary, map(bin, ba)))))


def count_zero(arr):
    arr = np.asarray(arr)
    return np.count_nonzero(arr == 0)


def assert_eq(left, right, **kwargs):
    """ Assert that two cudf-like things are equivalent

    This equality test works for pandas/cudf dataframes/series/indexes/scalars
    in the same way, and so makes it easier to perform parametrized testing
    without switching between assert_frame_equal/assert_series_equal/...
    functions.
    """
    __tracebackhide__ = True
    downcast = kwargs.pop('downcast', False)

    if hasattr(left, "to_pandas"):
        left = left.to_pandas()
        if downcast: 
            left = downcast_to_lowercase(left)
    if hasattr(right, "to_pandas"):
        right = right.to_pandas()
        if downcast:
            right = downcast_to_lowercase(right)
    if isinstance(left, cupy.ndarray):
        left = cupy.asnumpy(left)
    if isinstance(right, cupy.ndarray):
        right = cupy.asnumpy(right)
    if isinstance(left, pd.DataFrame):
        tm.assert_frame_equal(left, right, **kwargs)
    elif isinstance(left, pd.Series):
        tm.assert_series_equal(left, right, **kwargs)
    elif isinstance(left, pd.Index):
        tm.assert_index_equal(left, right, **kwargs)
    elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if np.issubdtype(left.dtype, np.floating) and np.issubdtype(
            right.dtype, np.floating
        ):
            assert np.allclose(left, right, equal_nan=True)
        else:
            assert np.array_equal(left, right)
    else:
        if left == right:
            return True
        else:
            if np.isnan(left):
                assert np.isnan(right)
            else:
                assert np.allclose(left, right, equal_nan=True)
    return True

def downcast_to_lowercase(obj):
    if isinstance(obj, pd.Series):
        if isinstance(obj.dtype, pd.core.dtypes.base.ExtensionDtype):
            obj = obj.astype(obj.dtype.type)
    elif isinstance(obj, pd.DataFrame):
        for col in obj.columns:
            if isinstance(obj[col].dtype, pd.core.dtypes.base.ExtensionDtype):
                obj[col] = obj[col].astype(obj[col].dtype.type)
    return obj
def assert_neq(left, right, **kwargs):
    __tracebackhide__ = True
    try:
        assert_eq(left, right, **kwargs)
    except AssertionError:
        pass
    else:
        raise AssertionError


def gen_rand(dtype, size, **kwargs):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        res = np.random.random(size=size).astype(dtype)
        if kwargs.get("positive_only", False):
            return res
        else:
            return res * 2 - 1
    elif dtype == np.int8 or dtype == np.int16:
        low = kwargs.get("low", -32)
        high = kwargs.get("high", 32)
        return np.random.randint(low=low, high=high, size=size).astype(dtype)
    elif dtype.kind == "i":
        low = kwargs.get("low", -10000)
        high = kwargs.get("high", 10000)
        return np.random.randint(low=low, high=high, size=size).astype(dtype)
    elif dtype == np.uint8 or dtype == np.uint16:
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 32)
        return np.random.randint(low=low, high=high, size=size).astype(dtype)
    elif dtype.kind == "u":
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 128)
        return np.random.randint(low=low, high=high, size=size).astype(dtype)
    elif dtype.kind == "b":
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 1)
        return np.random.randint(low=low, high=high, size=size).astype(np.bool)
    raise NotImplementedError("dtype.kind={}".format(dtype.kind))


def gen_rand_series(dtype, size, **kwargs):
    values = gen_rand(dtype, size, **kwargs)
    if kwargs.get("has_nulls", False):
        return cudf.Series.from_masked_array(values, random_bitmask(size))

    return cudf.Series(values)

def promote_to_pd_nullable_dtype(obj):
    mapping = {
        np.dtype('uint8'): pd.UInt8Dtype(),
        np.dtype('uint16'): pd.UInt16Dtype(),
        np.dtype('uint32'): pd.UInt32Dtype(),
        np.dtype('uint64'): pd.UInt64Dtype(),
        np.dtype('int8'): pd.Int8Dtype(),
        np.dtype('int16'): pd.Int16Dtype(),
        np.dtype('int32'): pd.Int32Dtype(),
        np.dtype('int64'): pd.Int64Dtype(),
        np.dtype('object'): pd.StringDtype(),
        np.dtype('bool'): pd.BooleanDtype(),
    }

    if isinstance(obj, pd.Series):
        dt = mapping.get(obj.dtype, obj.dtype)
        obj = obj.astype(dt)
        return obj
    elif isinstance(obj, pd.DataFrame):
        for colname in obj.columns:
            col = obj[colname]
            dt = mapping.get(col.dtype, col.dtype)
            obj[colname] = col.astype(dt)
        return obj
    else:
        return obj

@contextmanager
def does_not_raise():
    yield

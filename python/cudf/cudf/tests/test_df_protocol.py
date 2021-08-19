import datetime
import cupy
import numpy as np
import pytest
from cudf.core import df_protocol

import cudf
from cudf.testing import _utils as utils
from cudf.testing._utils import (
    ALL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    assert_eq,
    assert_exceptions_equal,
    does_not_raise,
    gen_rand,
)


def _from_dataframe_equals(df, copy=False):
    df2 = df_protocol._from_dataframe(df.__dataframe__(), copy=copy)
    assert_eq(df, df2)

def _from_dataframe_exception(df):
    exception_msg = "This operation must copy data from CPU to GPU. Set `copy=True` to allow it."
    with pytest.raises(TypeError, match=exception_msg):
        df2 = from_dataframe(df, copy=False)

def _datatype(data):
    cdf = cudf.DataFrame(data=data)
    _from_dataframe_equals(cdf, copy=False)
    _from_dataframe_equals(cdf, copy=True)

    
def test_int_dtype():
    data_int = dict(a=[1, 2, 3], b=[9, 10, 11])
    _datatype(data_int)

def test_float_dtype():
    data_float = dict(a=[1.5, 2.5, 3.5], b=[9.2, 10.5, 11.8])
    _datatype(data_float)

def test_mixed_intfloat_dtype():
    data_intfloat = dict(a=[1, 2, 3], b=[1.5, 2.5, 3.5])
    _datatype(data_intfloat)

def test_categorical_dtype():

    def test__dataframe__(df):
        # Some detailed testing for correctness of dtype:
        col = df.__dataframe__().get_column_by_name('A')
        assert col.dtype[0] == df_protocol._DtypeKind.CATEGORICAL
        assert col.null_count == 0
        assert col.num_chunks() == 1
        assert col.describe_categorical == (False, True, {0: 1, 1: 2, 2: 5})

    cdf = cudf.DataFrame({"A": [1, 2, 5, 1]})
    cdf["A"] = cdf["A"].astype("category")
    test__dataframe__(cdf)
    _from_dataframe_equals(cdf, copy=False)
    _from_dataframe_equals(cdf, copy=True)

# def test_bool_dtype():
#     data_bool = dict(a=[True, True, False], b=[False, True, False])
#     _datatype(data_bool)
import datetime
import cupy
import numpy as np
import pytest
from cudf.core.df_protocol import (
    _from_dataframe, 
    _DtypeKind,
    __dataframe__,
    _CuDFDataFrame
)

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
import pandas as pd


def _test_from_dataframe_equals(dfobj, copy=False):
    df2 = _from_dataframe(dfobj, copy=copy)

    if isinstance(dfobj._df, cudf.DataFrame):
        assert_eq(dfobj._df, df2)

    elif isinstance(dfobj._df, pd.DataFrame):
        assert_eq(cudf.DataFrame(dfobj._df), df2)

    else:
        raise TypeError(f"{type(dfobj._df)} not supported yet.")


def _test_from_dataframe_exception(dfobj):
    exception_msg = "This operation must copy data from CPU to GPU. Set `copy=True` to allow it."
    with pytest.raises(TypeError, match=exception_msg):
        df2 = _from_dataframe(dfobj, copy=False)

def _test_datatype(data):
    cdf = cudf.DataFrame(data=data)
    cdfobj = cdf.__dataframe__()
    print(cdfobj)
    _test_from_dataframe_equals(cdfobj, copy=False)
    _test_from_dataframe_equals(cdfobj, copy=True)

    # pdf = pd.DataFrame(data=data)
    # cpu_dfobj = _CuDFDataFrame(pdf)
    # _test_from_dataframe_exception(cpu_dfobj)
    # _test_from_dataframe_equals(cpu_dfobj, copy=True)
    

    
def test_int_dtype():
    data_int = dict(a=[1, 2, 3], b=[9, 10, 11])
    _test_datatype(data_int)

def test_float_dtype():
    data_float = dict(a=[1.5, 2.5, 3.5], b=[9.2, 10.5, 11.8])
    _test_datatype(data_float)

def test_mixed_intfloat_dtype():
    data_intfloat = dict(a=[1, 2, 3], b=[1.5, 2.5, 3.5])
    _test_datatype(data_intfloat)

def test_categorical_dtype():

    def test__dataframe__(df):
        # Some detailed testing for correctness of dtype:
        col = df.__dataframe__().get_column_by_name('A')
        assert col.dtype[0] == _DtypeKind.CATEGORICAL
        assert col.null_count == 0
        assert col.num_chunks() == 1
        assert col.describe_categorical == (False, True, {0: 1, 1: 2, 2: 5})

    cdf = cudf.DataFrame({"A": [1, 2, 5, 1]})
    cdf["A"] = cdf["A"].astype("category")
    test__dataframe__(cdf)
    _test_from_dataframe_equals(cdf.__dataframe__(), copy=False)
    _test_from_dataframe_equals(cdf.__dataframe__(), copy=True)

def test_NA_int_dtype():
    data_int = dict(a=[1, None, 3], b=[9, 10, None])
    _test_datatype(data_int)

# def test_NA2_int_dtype():
#     data_int = dict(a=[1, None, 3, None, 5], b=[9, 10, None, 7, 8])
#     _test_datatype(data_int)


# def test_bool_dtype():
#     data_bool = dict(a=[True, True, False], b=[False, True, False])
#     _datatype(data_bool)
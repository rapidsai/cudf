import cudf
from cudf.core.udf.pipeline import nulludf
from cudf.tests.utils import assert_eq, NUMERIC_TYPES
import pandas as pd
import itertools
import pytest
import operator

arith_ops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow

]

comparison_ops = [
    operator.eq, 
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge
]

def run_masked_udf_test(func_pdf, func_gdf, data, **kwargs):
    gdf = data
    pdf = data.to_pandas(nullable=True)

    expect = pdf.apply(
        lambda row: func_pdf(
            *[row[i] for i in data.columns]
        ),
        axis=1
    )
    obtain = gdf.apply(
        lambda row: func_gdf(
            *[row[i] for i in data.columns]
            ),
            axis=1
    )
    assert_eq(expect, obtain, **kwargs)

@pytest.mark.parametrize('op', arith_ops)
def test_arith_masked_vs_masked(op):
    # This test should test all the typing
    # and lowering for arithmetic ops between
    # two columns 
    def func_pdf(x, y):
        return op(x, y)

    @nulludf
    def func_gdf(x, y):
        return op(x, y)

    gdf = cudf.DataFrame({
        'a':[1,None,3, None],
        'b':[4,5,None, None]
    })
    run_masked_udf_test(func_pdf, func_gdf, gdf, check_dtype=False)

@pytest.mark.parametrize('op', comparison_ops)
def test_compare_masked_vs_masked(op):
    # this test should test all the 
    # typing and lowering for comparisons 
    # between columns

    def func_pdf(x, y):
        return op(x, y)

    @nulludf
    def func_gdf(x, y):
        return op(x, y)

    # we should get:
    # [?, ?, <NA>, <NA>, <NA>]
    gdf = cudf.DataFrame({
        'a': [1, 0, None, 1, None],
        'b': [0, 1, 0, None, None]
    })
    run_masked_udf_test(func_pdf, func_gdf, gdf, check_dtype=False)

@pytest.mark.parametrize('op', arith_ops)
@pytest.mark.parametrize('constant', [1, 1.5])
def test_arith_masked_vs_constant(op, constant):
    def func_pdf(x):
        return op(x, constant)
    
    @nulludf
    def func_gdf(x):
        return op(x, constant)

    # Just a single column -> result will be all NA
    gdf = cudf.DataFrame({
        'data': [1,2,None]
    })

    run_masked_udf_test(func_pdf, func_gdf, gdf, check_dtype=False)

@pytest.mark.parametrize('op', comparison_ops)
@pytest.mark.parametrize('constant', [1, 1.5])
def test_compare_masked_vs_constant(op, constant):
    '''
    technically the exact same test as above
    '''
    def func_pdf(x):
        return op(x, constant)
    
    @nulludf
    def func_gdf(x):
        return op(x, constant)

    # Just a single column -> result will be all NA
    gdf = cudf.DataFrame({
        'data': [1,2,None]
    })
    run_masked_udf_test(func_pdf, func_gdf, gdf, check_dtype=False)



@pytest.mark.parametrize('op', arith_ops)
def test_arith_masked_vs_null(op):
    def func_pdf(x):
        return op(x, pd.NA)

    @nulludf
    def func_gdf(x):
        return op(x, cudf.NA)

    gdf = cudf.DataFrame({
        'data': [1, None, 3]
    })
    run_masked_udf_test(func_pdf, func_gdf, gdf, check_dtype=False)

@pytest.mark.parametrize('op', arith_ops)
def test_arith_masked_vs_null_reflected(op):
    def func_pdf(x):
        return op(pd.NA, x)

    @nulludf
    def func_gdf(x):
        return op(cudf.NA, x)

    gdf = cudf.DataFrame({
        'data': [1, None, 3]
    })
    run_masked_udf_test(func_pdf, func_gdf, gdf, check_dtype=False)


def test_masked_is_null_conditional():
    def func_pdf(x, y):
        if x is pd.NA:
            return y
        else:
            return x + y

    @nulludf
    def func_gdf(x, y):
        if x is cudf.NA:
            return y
        else:
            return x + y


    gdf = cudf.DataFrame({
        'a':[1,None,3, None],
        'b':[4,5,None, None]
    })
    run_masked_udf_test(func_pdf, func_gdf, gdf, check_dtype=False)


@pytest.mark.parametrize('dtype_a', list(NUMERIC_TYPES))
@pytest.mark.parametrize('dtype_b', list(NUMERIC_TYPES))
def test_apply_mixed_dtypes(dtype_a, dtype_b):
    def func_pdf(x, y):
        return x + y
    
    @nulludf
    def func_gdf(x, y):
        return x + y

    gdf = cudf.DataFrame({
        'a':[1.5,None,3, None],
        'b':[4,5,None, None]
    })
    gdf['a'] = gdf['a'].astype(dtype_a)
    gdf['b'] = gdf['b'].astype(dtype_b)

    run_masked_udf_test(func_pdf, func_gdf, gdf, check_dtype=False) 


@pytest.mark.parametrize('val', [
    5, 5.5
])
def test_apply_return_literal(val):
    def func_pdf(x, y):
        if x is pd.NA:
            return val
        else:
            return x + y

    @nulludf
    def func_gdf(x, y):
        if x is cudf.NA:
            return val # Masked(5, True)
        else:
            return x + y


    gdf = cudf.DataFrame({
        'a':[1,None,3, None],
        'b':[4,5,None, None]
    })

    run_masked_udf_test(func_pdf, func_gdf, gdf, check_dtype=False) 

def test_apply_return_null():
    '''
    Tests casting / unification of Masked and NA
    '''
    def func_pdf(x):
        if x is pd.NA:
            return pd.NA
        else:
            return x 

    @nulludf
    def func_gdf(x):
        if x is cudf.NA:
            return cudf.NA
        else:
            return x

    gdf = cudf.DataFrame({'a': [1, None, 3]})
    run_masked_udf_test(func_pdf, func_gdf, gdf, check_dtype=False) 

def test_apply_return_either_null_or_literal():
    def func_pdf(x):
        if x > 5:
            return 2
        else:
            return pd.NA

    @nulludf
    def func_gdf(x):
        if x > 5:
            return 2
        else:
            return cudf.NA

    gdf = cudf.DataFrame({'a': [1, 3, 6]})
    run_masked_udf_test(func_pdf, func_gdf, gdf, check_dtype=False) 

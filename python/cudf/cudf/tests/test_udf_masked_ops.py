import cudf
from cudf.core.udf import nulludf
from cudf.tests.utils import assert_eq
import pandas as pd


def test_apply_basic():
    def func_pdf(x, y):
        return x + y

    @nulludf
    def func_gdf(x, y):
        return x + y


    gdf = cudf.DataFrame({
        'a':[1,2,3],
        'b':[4,5,6]
    })

    pdf = gdf.to_pandas()

    expect = pdf.apply(lambda row: func_pdf(row['a'], row['b']), axis=1)
    obtain = gdf.apply(lambda row: func_gdf(row['a'], row['b']), axis=1)

    assert_eq(expect, obtain)

def test_apply_null():
    def func_pdf(x, y):
        return x + y

    @nulludf
    def func_gdf(x, y):
        return x + y


    gdf = cudf.DataFrame({
        'a':[1,None,3, None],
        'b':[4,5,None, None]
    })

    pdf = gdf.to_pandas()

    expect = pdf.apply(lambda row: func_pdf(row['a'], row['b']), axis=1)
    obtain = gdf.apply(lambda row: func_gdf(row['a'], row['b']), axis=1)

    assert_eq(expect, obtain)

def test_apply_NA_conditional():
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

    pdf = gdf.to_pandas()

    expect = pdf.apply(lambda row: func_pdf(row['a'], row['b']), axis=1)
    obtain = gdf.apply(lambda row: func_gdf(row['a'], row['b']), axis=1)

    assert_eq(expect, obtain)


def test_apply_return_literal():
    # 1. Casting rule literal -> Masked
    #  -> a) make it so numba knows that we can even promote literals to Masked ()
    #  -> b) implement custom lowering to specify how this actually happens (python only)


    # 2. Custom unfication code


    # numba/core/type
    def func_pdf(x, y):
        if x is pd.NA:
            return 5
        else:
            return x + y

    @nulludf
    def func_gdf(x, y):
        if x is cudf.NA:
            return 5 # Masked(5, True)
        else:
            return x + y


    gdf = cudf.DataFrame({
        'a':[1,None,3, None],
        'b':[4,5,None, None]
    })

    pdf = gdf.to_pandas()

    expect = pdf.apply(lambda row: func_pdf(row['a'], row['b']), axis=1)
    obtain = gdf.apply(lambda row: func_gdf(row['a'], row['b']), axis=1)

    assert_eq(expect, obtain)

import operator

import numpy as np
import pandas as pd
import pytest

import dask.dataframe as dd

import cudf


def _make_empty_frame(npartitions=2):
    df = pd.DataFrame({"x": [], "y": []})
    gdf = cudf.DataFrame.from_pandas(df)
    dgf = dd.from_pandas(gdf, npartitions=npartitions)
    return dgf


def _make_random_frame(nelem, npartitions=2):
    df = pd.DataFrame(
        {"x": np.random.random(size=nelem), "y": np.random.random(size=nelem)}
    )
    gdf = cudf.DataFrame.from_pandas(df)
    dgf = dd.from_pandas(gdf, npartitions=npartitions)
    return df, dgf


def _make_random_frame_float(nelem, npartitions=2):
    df = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=nelem),
            "y": np.random.normal(size=nelem) + 1,
        }
    )
    gdf = cudf.from_pandas(df)
    dgf = dd.from_pandas(gdf, npartitions=npartitions)
    return df, dgf


_binops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
    operator.lt,
    operator.le,
]

_cudf_bind_ops = ["add", "sub", "mul", "truediv", "floordiv", "mod", "pow"]


@pytest.mark.parametrize("binop", _binops)
def test_series_binops_integer(binop):
    np.random.seed(0)
    size = 1000
    lhs_df, lhs_gdf = _make_random_frame(size)
    rhs_df, rhs_gdf = _make_random_frame(size)
    got = binop(lhs_gdf.x, rhs_gdf.y)
    exp = binop(lhs_df.x, rhs_df.y)
    dd.assert_eq(got, exp)


@pytest.mark.parametrize("binop", _binops)
def test_series_binops_float(binop):
    np.random.seed(0)
    size = 1000
    lhs_df, lhs_gdf = _make_random_frame_float(size)
    rhs_df, rhs_gdf = _make_random_frame_float(size)
    got = binop(lhs_gdf.x, rhs_gdf.y)
    exp = binop(lhs_df.x, rhs_df.y)
    dd.assert_eq(got, exp)


@pytest.mark.parametrize("operator", _cudf_bind_ops)
def test_df_series_bind_ops(operator):
    np.random.seed(0)
    size = 1000
    lhs_df, lhs_gdf = _make_random_frame_float(size)
    rhs = np.random.rand()

    got = getattr(lhs_gdf, operator)(rhs)
    exp = getattr(lhs_df, operator)(rhs)

    dd.assert_eq(got, exp)

    for col in lhs_gdf.columns:
        got = getattr(lhs_gdf[col], operator)(rhs)
        exp = getattr(lhs_df[col], operator)(rhs)
        dd.assert_eq(got, exp)

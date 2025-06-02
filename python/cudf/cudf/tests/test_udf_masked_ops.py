# Copyright (c) 2021-2025, NVIDIA CORPORATION.
import math
import operator

import numpy as np
import pytest
from numba import cuda

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.core.missing import NA
from cudf.core.udf._ops import (
    arith_ops,
    bitwise_ops,
    comparison_ops,
    unary_ops,
)
from cudf.core.udf.api import Masked
from cudf.core.udf.utils import precompiled
from cudf.testing import assert_eq
from cudf.testing._utils import (
    _decimal_series,
    parametrize_numeric_dtypes_pairwise,
    sv_to_managed_udf_str,
)


@pytest.fixture(scope="module")
def str_udf_data():
    return cudf.DataFrame(
        {
            "str_col": [
                "abc",
                "ABC",
                "AbC",
                "123",
                "123aBc",
                "123@.!",
                "",
                "rapids ai",
                "gpu",
                "True",
                "False",
                "1.234",
                ".123a",
                "0.013",
                "1.0",
                "01",
                "20010101",
                "cudf",
                "cuda",
                "gpu",
                "This Is A Title",
                "This is Not a Title",
                "Neither is This a Title",
                "NoT a TiTlE",
                "123 Title Works",
            ]
        }
    )


@pytest.fixture(params=["a", "cu", "2", "gpu", "", " "])
def substr(request):
    return request.param


def run_masked_udf_test(func, data, args=(), nullable=True, **kwargs):
    gdf = data
    pdf = data.to_pandas(nullable=nullable)

    expect = pdf.apply(func, args=args, axis=1)
    obtain = gdf.apply(func, args=args, axis=1)
    assert_eq(expect, obtain, **kwargs)


def run_masked_string_udf_test(func, data, args=(), **kwargs):
    gdf = data
    pdf = data.to_pandas(nullable=True)

    def row_wrapper(row):
        st = row["str_col"]
        return func(st)

    expect = pdf.apply(row_wrapper, args=args, axis=1)

    func = cuda.jit(device=True)(func)
    obtain = gdf.apply(row_wrapper, args=args, axis=1)
    assert_eq(expect, obtain, **kwargs)

    # strings that come directly from input columns are backed by
    # MaskedType(string_view) types. But new strings that are returned
    # from functions or operators are backed by MaskedType(udf_string)
    # types. We need to make sure all of our methods work on both kind
    # of MaskedType. This function promotes the former to the latter
    # prior to running the input function
    def udf_string_wrapper(row):
        masked_udf_str = Masked(
            sv_to_managed_udf_str(row["str_col"].value), row["str_col"].valid
        )
        return func(masked_udf_str)

    obtain = gdf.apply(udf_string_wrapper, args=args, axis=1)
    assert_eq(expect, obtain, **kwargs)


def run_masked_udf_series(func, data, args=(), **kwargs):
    gsr = data
    psr = data.to_pandas(nullable=True)

    expect = psr.apply(func, args=args)
    obtain = gsr.apply(func, args=args)
    assert_eq(expect, obtain, **kwargs)


@pytest.mark.parametrize("op", arith_ops)
def test_arith_masked_vs_masked(op):
    # This test should test all the typing
    # and lowering for arithmetic ops between
    # two columns
    def func(row):
        x = row["a"]
        y = row["b"]
        return op(x, y)

    gdf = cudf.DataFrame({"a": [1, None, 3, None], "b": [4, 5, None, None]})
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("op", bitwise_ops)
def test_bitwise_masked_vs_masked(op):
    # This test should test all the typing
    # and lowering for bitwise ops between
    # two columns
    def func(row):
        x = row["a"]
        y = row["b"]
        return op(x, y)

    gdf = cudf.DataFrame(
        {
            "a": [1, 0, 1, 0, 0b1011, 42, None],
            "b": [1, 1, 0, 0, 0b1100, -42, 5],
        }
    )
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize(
    "dtype_l",
    ["datetime64[ns]", "datetime64[us]", "datetime64[ms]", "datetime64[s]"],
)
@pytest.mark.parametrize(
    "dtype_r",
    [
        "timedelta64[ns]",
        "timedelta64[us]",
        "timedelta64[ms]",
        "timedelta64[s]",
        "datetime64[ns]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[s]",
    ],
)
@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_arith_masked_vs_masked_datelike(op, dtype_l, dtype_r):
    # Datetime version of the above
    # does not test all dtype combinations for now
    if "datetime" in dtype_l and "datetime" in dtype_r and op is operator.add:
        # don't try adding datetimes to datetimes.
        pytest.skip("Adding datetime to datetime is not valid")

    def func(row):
        x = row["a"]
        y = row["b"]
        return op(x, y)

    gdf = cudf.DataFrame(
        {
            "a": ["2011-01-01", cudf.NA, "2011-03-01", cudf.NA],
            "b": [4, 5, cudf.NA, cudf.NA],
        }
    )
    gdf["a"] = gdf["a"].astype(dtype_l)
    gdf["b"] = gdf["b"].astype(dtype_r)

    pdf = gdf.to_pandas()
    expect = op(pdf["a"], pdf["b"])
    obtain = gdf.apply(func, axis=1)
    assert_eq(expect, obtain, check_dtype=False)
    # TODO: After the following pandas issue is
    # fixed, uncomment the following line and delete
    # through `to_pandas()` statement.
    # https://github.com/pandas-dev/pandas/issues/52411

    # run_masked_udf_test(func, gdf, nullable=False, check_dtype=False)


@pytest.mark.parametrize("op", comparison_ops)
def test_compare_masked_vs_masked(op):
    # this test should test all the
    # typing and lowering for comparisons
    # between columns

    def func(row):
        x = row["a"]
        y = row["b"]
        return op(x, y)

    # we should get:
    # [?, ?, <NA>, <NA>, <NA>]
    gdf = cudf.DataFrame(
        {"a": [1, 0, None, 1, None], "b": [0, 1, 0, None, None]}
    )
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("op", arith_ops)
@pytest.mark.parametrize("constant", [1, 1.5, True, False])
@pytest.mark.parametrize("data", [[1, 2, cudf.NA]])
def test_arith_masked_vs_constant(op, constant, data):
    def func(row):
        x = row["data"]
        return op(x, constant)

    gdf = cudf.DataFrame({"data": data})

    if constant is False and op in {
        operator.mod,
        operator.pow,
        operator.truediv,
        operator.floordiv,
        operator.imod,
        operator.ipow,
        operator.itruediv,
        operator.ifloordiv,
    }:
        # The following tests cases yield undefined behavior:
        # - truediv(x, False) because its dividing by zero
        # - floordiv(x, False) because its dividing by zero
        # - mod(x, False) because its mod by zero,
        # - pow(x, False) because we have an NA in the series and pandas
        #   insists that (NA**0 == 1) where we do not
        pytest.skip()
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("op", arith_ops)
@pytest.mark.parametrize("constant", [1, 1.5, True, False])
@pytest.mark.parametrize("data", [[2, 3, cudf.NA], [1, cudf.NA, 1]])
def test_arith_masked_vs_constant_reflected(request, op, constant, data):
    def func(row):
        x = row["data"]
        return op(constant, x)

    # Just a single column -> result will be all NA
    gdf = cudf.DataFrame({"data": data})

    # cudf differs from pandas for 1**NA
    request.applymarker(
        pytest.mark.xfail(
            condition=(constant == 1 and op in {operator.pow, operator.ipow}),
            reason="https://github.com/rapidsai/cudf/issues/7478",
        )
    )
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("op", arith_ops)
@pytest.mark.parametrize("data", [[1, cudf.NA, 3], [2, 3, cudf.NA]])
def test_arith_masked_vs_null(request, op, data):
    def func(row):
        x = row["data"]
        return op(x, NA)

    gdf = cudf.DataFrame({"data": data})

    # In pandas, 1**NA == 1.
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                (gdf["data"] == 1).any()
                and op in {operator.pow, operator.ipow}
            ),
            reason="https://github.com/rapidsai/cudf/issues/7478",
        )
    )
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("op", arith_ops)
def test_arith_masked_vs_null_reflected(op):
    def func(row):
        x = row["data"]
        return op(NA, x)

    gdf = cudf.DataFrame({"data": [1, None, 3]})
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("op", unary_ops)
def test_unary_masked(op):
    # This test should test all the typing
    # and lowering for unary ops

    def func(row):
        x = row["a"]
        return op(x) if x is not NA else NA

    if "log" in op.__name__:
        gdf = cudf.DataFrame({"a": [0.1, 1.0, None, 3.5, 1e8]})
    elif op.__name__ in {"asin", "acos"}:
        gdf = cudf.DataFrame({"a": [0.0, 0.5, None, 1.0]})
    elif op.__name__ in {"atanh"}:
        gdf = cudf.DataFrame({"a": [0.0, -0.5, None, 0.8]})
    elif op.__name__ in {"acosh", "sqrt", "lgamma"}:
        gdf = cudf.DataFrame({"a": [1.0, 2.0, None, 11.0]})
    elif op.__name__ in {"gamma"}:
        gdf = cudf.DataFrame({"a": [0.1, 2, None, 4]})
    elif op.__name__ in {"invert"}:
        gdf = cudf.DataFrame({"a": [-100, 128, None, 0]}, dtype="int64")
    else:
        gdf = cudf.DataFrame({"a": [-125.60, 395.2, 0.0, None]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_masked_is_null_conditional():
    def func(row):
        x = row["a"]
        y = row["b"]
        if x is NA:
            return y
        else:
            return x + y

    gdf = cudf.DataFrame({"a": [1, None, 3, None], "b": [4, 5, None, None]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_apply_contains():
    def func(row):
        x = row["a"]
        return x in [1, 2]

    gdf = cudf.DataFrame({"a": [1, 3]})
    run_masked_udf_test(func, gdf, check_dtype=False)


@parametrize_numeric_dtypes_pairwise
@pytest.mark.parametrize("op", [operator.add, operator.and_, operator.eq])
def test_apply_mixed_dtypes(left_dtype, right_dtype, op):
    """
    Test that operations can be performed between columns
    of different dtypes and return a column with the correct
    values and nulls
    """

    # First perform the op on two dummy data on host, if numpy can
    # safely type cast, we should expect it to work in udf too.
    try:
        op(np.dtype(left_dtype).type(0), np.dtype(right_dtype).type(42))
    except TypeError:
        pytest.skip("Operation is unsupported for corresponding dtype.")

    def func(row):
        x = row["a"]
        y = row["b"]
        return op(x, y)

    gdf = cudf.DataFrame({"a": [1.5, None, 3, None], "b": [4, 5, None, None]})
    gdf["a"] = gdf["a"].astype(left_dtype)
    gdf["b"] = gdf["b"].astype(right_dtype)

    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("val", [5, 5.5])
def test_apply_return_literal(val):
    """
    Test unification codepath for scalars and MaskedType
    makes sure that numba knows how to cast a scalar value
    to a MaskedType
    """

    def func(row):
        x = row["a"]
        y = row["b"]
        if x is not NA and x < 2:
            return val
        else:
            return x + y

    gdf = cudf.DataFrame({"a": [1, None, 3, None], "b": [4, 5, None, None]})

    run_masked_udf_test(func, gdf, check_dtype=False)


def test_apply_return_null():
    """
    Tests casting / unification of Masked and NA
    """

    def func(row):
        x = row["a"]
        if x is NA:
            return NA
        else:
            return x

    gdf = cudf.DataFrame({"a": [1, None, 3]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_apply_return_either_null_or_literal():
    def func(row):
        x = row["a"]
        if x > 5:
            return 2
        else:
            return NA

    gdf = cudf.DataFrame({"a": [1, 3, 6]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_apply_return_literal_only():
    def func(x):
        return 5

    gdf = cudf.DataFrame({"a": [1, None, 3]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_apply_everything():
    def func(row):
        w = row["a"]
        x = row["b"]
        y = row["c"]
        z = row["d"]
        if x is NA:
            return w + y - z
        elif ((z > y) is not NA) and z > y:
            return x
        elif ((x + y) is not NA) and x + y == 0:
            return z / x
        elif x + y is NA:
            return 2.5
        elif w > 100:
            return (
                math.sin(x)
                + math.sqrt(y)
                - (-z)
                + math.lgamma(x) * math.fabs(-0.8) / math.radians(3.14)
            )
        else:
            return y > 2

    gdf = cudf.DataFrame(
        {
            "a": [1, 3, 6, 0, None, 5, None, 101],
            "b": [3.0, 2.5, None, 5.0, 1.0, 5.0, 11.0, 1.0],
            "c": [2, 3, 6, 0, None, 5, None, 6],
            "d": [4, None, 6, 0, None, 5, None, 7.5],
        }
    )
    run_masked_udf_test(func, gdf, check_dtype=False)


###


@pytest.mark.parametrize(
    "data,name",
    [([1, 2, 3], None), ([1, cudf.NA, 3], None), ([1, 2, 3], "test_name")],
)
def test_series_apply_basic(data, name):
    data = cudf.Series(data, name=name)

    def func(x):
        return x + 1

    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.xfail(
    PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="https://github.com/pandas-dev/pandas/issues/57390",
)
def test_series_apply_null_conditional():
    def func(x):
        if x is NA:
            return 42
        else:
            return x - 1

    data = cudf.Series([1, cudf.NA, 3])

    run_masked_udf_series(func, data)


###


@pytest.mark.parametrize("op", arith_ops)
def test_series_arith_masked_vs_masked(op):
    def func(x):
        return op(x, x)

    data = cudf.Series([1, cudf.NA, 3])
    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.xfail(
    PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="https://github.com/pandas-dev/pandas/issues/57390",
)
@pytest.mark.parametrize("op", comparison_ops)
def test_series_compare_masked_vs_masked(op):
    """
    In the series case, only one other MaskedType to compare with
    - itself
    """

    def func(x):
        return op(x, x)

    data = cudf.Series([1, cudf.NA, 3])
    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.parametrize("op", arith_ops)
@pytest.mark.parametrize("constant", [1, 1.5, cudf.NA])
def test_series_arith_masked_vs_constant(request, op, constant):
    def func(x):
        return op(x, constant)

    # Just a single column -> result will be all NA
    data = cudf.Series([1, 2, cudf.NA])
    # in pandas, 1**NA == 1. In cudf, 1**NA == NA.
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                constant is cudf.NA and op in {operator.pow, operator.ipow}
            ),
            reason="https://github.com/rapidsai/cudf/issues/7478",
        )
    )
    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.parametrize("op", arith_ops)
@pytest.mark.parametrize("constant", [1, 1.5, cudf.NA])
def test_series_arith_masked_vs_constant_reflected(request, op, constant):
    def func(x):
        return op(constant, x)

    # Just a single column -> result will be all NA
    data = cudf.Series([1, 2, cudf.NA])
    # Using in {1} since bool(NA == 1) raises a TypeError since NA is
    # neither truthy nor falsy
    # in pandas, 1**NA == 1. In cudf, 1**NA == NA.
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                constant in {1} and op in {operator.pow, operator.ipow}
            ),
            reason="https://github.com/rapidsai/cudf/issues/7478",
        )
    )
    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.xfail(
    PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="https://github.com/pandas-dev/pandas/issues/57390",
)
def test_series_masked_is_null_conditional():
    def func(x):
        if x is NA:
            return 42
        else:
            return x

    data = cudf.Series([1, cudf.NA, 3, cudf.NA])

    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.parametrize("op", arith_ops + comparison_ops)
def test_masked_udf_lambda_support(op):
    func = lambda row: op(row["a"], row["b"])  # noqa: E731

    data = cudf.DataFrame(
        {"a": [1, cudf.NA, 3, cudf.NA], "b": [1, 2, cudf.NA, cudf.NA]}
    )

    run_masked_udf_test(func, data, check_dtype=False)


@pytest.mark.parametrize("op", arith_ops + comparison_ops)
def test_masked_udf_nested_function_support(op):
    """
    Nested functions need to be explicitly jitted by the user
    for numba to recognize them. Unfortunately the object
    representing the jitted function can not itself be used in
    pandas udfs.
    """

    def inner(x, y):
        return op(x, y)

    def outer(row):
        x = row["a"]
        y = row["b"]
        return inner(x, y)

    gdf = cudf.DataFrame(
        {"a": [1, cudf.NA, 3, cudf.NA], "b": [1, 2, cudf.NA, cudf.NA]}
    )

    with pytest.raises(ValueError):
        gdf.apply(outer, axis=1)

    pdf = gdf.to_pandas(nullable=True)
    inner_gpu = cuda.jit(device=True)(inner)

    def outer_gpu(row):
        x = row["a"]
        y = row["b"]
        return inner_gpu(x, y)

    got = gdf.apply(outer_gpu, axis=1)
    expect = pdf.apply(outer, axis=1)
    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
        {"a": [1, 2, 3], "c": [4, 5, 6], "b": [7, 8, 9]},
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": ["a", "b", "c"]},
    ],
)
def test_masked_udf_subset_selection(data):
    def func(row):
        return row["a"] + row["b"]

    data = cudf.DataFrame(data)
    run_masked_udf_test(func, data)


@pytest.mark.parametrize(
    "unsupported_col",
    [
        _decimal_series(
            ["1.0", "2.0", "3.0"], dtype=cudf.Decimal64Dtype(2, 1)
        ),
        cudf.Series([1, 2, 3], dtype="category"),
        cudf.interval_range(start=0, end=3),
        [[1, 2], [3, 4], [5, 6]],
        [{"a": 1}, {"a": 2}, {"a": 3}],
    ],
)
def test_masked_udf_unsupported_dtype(unsupported_col):
    data = cudf.DataFrame()
    data["unsupported_col"] = unsupported_col

    def func(row):
        return row["unsupported_col"]

    # check that we fail when an unsupported type is used within a function
    with pytest.raises(ValueError):
        data.apply(func, axis=1)

    # also check that a DF containing unsupported dtypes can still run a
    # function that does NOT involve any of the unsupported dtype columns
    data["supported_col"] = 1

    def other_func(row):
        return row["supported_col"]

    expect = cudf.Series(np.ones(len(data)))
    got = data.apply(other_func, axis=1)

    assert_eq(expect, got, check_dtype=False)


# tests for `DataFrame.apply(f, args=(x,y,z))`
# testing the whole space of possibilities is intractable
# these test the most rudimentary guaranteed functionality
@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, cudf.NA, 3]},
        {"a": [0.5, 2.0, cudf.NA, cudf.NA, 5.0]},
        {"a": [True, False, cudf.NA]},
    ],
)
@pytest.mark.parametrize("op", arith_ops + comparison_ops)
def test_masked_udf_scalar_args_binops(data, op):
    data = cudf.DataFrame(data)

    def func(row, c):
        return op(row["a"], c)

    run_masked_udf_test(func, data, args=(1,), check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, cudf.NA, 3]},
        {"a": [0.5, 2.0, cudf.NA, cudf.NA, 5.0]},
        {"a": [True, False, cudf.NA]},
    ],
)
@pytest.mark.parametrize("op", arith_ops + comparison_ops)
def test_masked_udf_scalar_args_binops_multiple(data, op):
    data = cudf.DataFrame(data)

    def func(row, c, k):
        x = op(row["a"], c)
        y = op(x, k)
        return y

    run_masked_udf_test(func, data, args=(1, 2), check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        [1, cudf.NA, 3],
        [0.5, 2.0, cudf.NA, cudf.NA, 5.0],
        [True, False, cudf.NA],
    ],
)
@pytest.mark.parametrize("op", arith_ops + comparison_ops)
def test_mask_udf_scalar_args_binops_series(data, op):
    data = cudf.Series(data)

    def func(x, c):
        return x + c

    run_masked_udf_series(func, data, args=(1,), check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        [1, cudf.NA, 3],
        [0.5, 2.0, cudf.NA, cudf.NA, 5.0],
        [True, False, cudf.NA],
    ],
)
@pytest.mark.parametrize("op", arith_ops + comparison_ops)
def test_masked_udf_scalar_args_binops_multiple_series(request, data, op):
    data = cudf.Series(data)
    request.applymarker(
        pytest.mark.xfail(
            op in comparison_ops
            and PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION
            and data.dtype.kind != "b",
            reason="https://github.com/pandas-dev/pandas/issues/57390",
        )
    )

    def func(data, c, k):
        x = op(data, c)
        y = op(x, k)
        return y

    run_masked_udf_series(func, data, args=(1, 2), check_dtype=False)


def test_masked_udf_caching():
    # Make sure similar functions that differ
    # by simple things like constants actually
    # recompile

    data = cudf.Series([1, 2, 3])

    expect = data**2
    got = data.apply(lambda x: x**2)
    assert_eq(expect, got, check_dtype=False)

    # update the constant value being used and make sure
    # it does not result in a cache hit

    expect = data**3
    got = data.apply(lambda x: x**3)
    assert_eq(expect, got, check_dtype=False)

    # make sure we get a hit when reapplying
    def f(x):
        return x + 1

    precompiled.clear()
    assert precompiled.currsize == 0
    data.apply(f)

    assert precompiled.currsize == 1
    data.apply(f)

    assert precompiled.currsize == 1

    # validate that changing the type of a scalar arg
    # results in a miss
    precompiled.clear()

    def f(x, c):
        return x + c

    data.apply(f, args=(1,))
    assert precompiled.currsize == 1

    data.apply(f, args=(1.5,))
    assert precompiled.currsize == 2


@pytest.mark.parametrize(
    "data", [[1.0, 0.0, 1.5], [1, 0, 2], [True, False, True]]
)
@pytest.mark.parametrize("operator", [float, int, bool])
def test_masked_udf_casting(operator, data):
    data = cudf.Series(data)

    def func(x):
        return operator(x)

    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        np.array(
            [0, 1, -1, 0, np.iinfo("int64").min, np.iinfo("int64").max],
            dtype="int64",
        ),
        np.array([0, 0, 1, np.iinfo("uint64").max], dtype="uint64"),
        np.array(
            [
                0,
                0.0,
                -1.0,
                1.5,
                -1.5,
                np.finfo("float64").min,
                np.finfo("float64").max,
                np.nan,
                np.inf,
                -np.inf,
            ],
            dtype="float64",
        ),
        [False, True, False, cudf.NA],
    ],
)
def test_masked_udf_abs(data):
    data = cudf.Series(data)
    data[0] = cudf.NA

    def func(x):
        return abs(x)

    run_masked_udf_series(func, data, check_dtype=False)


class TestStringUDFs:
    def test_string_udf_len(self, str_udf_data):
        def func(row):
            return len(row["str_col"])

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_startswith(self, str_udf_data, substr):
        def func(row):
            return row["str_col"].startswith(substr)

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_endswith(self, str_udf_data, substr):
        def func(row):
            return row["str_col"].endswith(substr)

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_find(self, str_udf_data, substr):
        def func(row):
            return row["str_col"].find(substr)

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_rfind(self, str_udf_data, substr):
        def func(row):
            return row["str_col"].rfind(substr)

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_contains(self, str_udf_data, substr):
        def func(row):
            return substr in row["str_col"]

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    @pytest.mark.parametrize("other", ["cudf", "123", "", " "])
    @pytest.mark.parametrize("cmpop", comparison_ops)
    def test_string_udf_cmpops(self, str_udf_data, other, cmpop):
        def func(row):
            return cmpop(row["str_col"], other)

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_isalnum(self, str_udf_data):
        def func(row):
            return row["str_col"].isalnum()

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_isalpha(self, str_udf_data):
        def func(row):
            return row["str_col"].isalpha()

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_isdigit(self, str_udf_data):
        def func(row):
            return row["str_col"].isdigit()

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_isdecimal(self, str_udf_data):
        def func(row):
            return row["str_col"].isdecimal()

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_isupper(self, str_udf_data):
        def func(row):
            return row["str_col"].isupper()

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_islower(self, str_udf_data):
        def func(row):
            return row["str_col"].islower()

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_isspace(self, str_udf_data):
        def func(row):
            return row["str_col"].isspace()

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_istitle(self, str_udf_data):
        def func(row):
            return row["str_col"].istitle()

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_count(self, str_udf_data, substr):
        def func(row):
            return row["str_col"].count(substr)

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    @pytest.mark.xfail(reason="Identity function not supported.")
    def test_string_udf_return_string(self, str_udf_data):
        def func(row):
            return row["str_col"]

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    @pytest.mark.parametrize("strip_char", ["1", "a", "12", " ", "", ".", "@"])
    def test_string_udf_strip(self, str_udf_data, strip_char):
        def func(row):
            return row["str_col"].strip(strip_char)

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    @pytest.mark.parametrize("strip_char", ["1", "a", "12", " ", "", ".", "@"])
    def test_string_udf_lstrip(self, str_udf_data, strip_char):
        def func(row):
            return row["str_col"].lstrip(strip_char)

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    @pytest.mark.parametrize("strip_char", ["1", "a", "12", " ", "", ".", "@"])
    def test_string_udf_rstrip(self, str_udf_data, strip_char):
        def func(row):
            return row["str_col"].rstrip(strip_char)

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_upper(self, str_udf_data):
        def func(row):
            return row["str_col"].upper()

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    def test_string_udf_lower(self, str_udf_data):
        def func(row):
            return row["str_col"].lower()

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    @pytest.mark.parametrize(
        "concat_char", ["1", "a", "12", " ", "", ".", "@"]
    )
    def test_string_udf_concat(self, str_udf_data, concat_char):
        def func(row):
            return row["str_col"] + concat_char

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

    @pytest.mark.parametrize("to_replace", ["a", "1", "", "@"])
    @pytest.mark.parametrize("replacement", ["a", "1", "", "@"])
    def test_string_udf_replace(self, str_udf_data, to_replace, replacement):
        def func(row):
            return row["str_col"].replace(to_replace, replacement)

        run_masked_udf_test(func, str_udf_data, check_dtype=False)

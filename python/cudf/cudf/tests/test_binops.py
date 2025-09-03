# Copyright (c) 2018-2025, NVIDIA CORPORATION.

import datetime
import decimal
import itertools
import operator
import re
import warnings
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import Index, Series
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_GE_220,
    PANDAS_VERSION,
)
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.core.column.column import as_column
from cudf.testing import _utils as utils, assert_eq
from cudf.utils.dtypes import (
    DATETIME_TYPES,
    FLOAT_TYPES,
    INTEGER_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
)

STRING_TYPES = {"str"}
pytest_xfail = pytest.mark.xfail
pytestmark = pytest.mark.spilling

# If spilling is enabled globally, we skip many test permutations
# to reduce running time.
if get_global_manager() is not None:
    DATETIME_TYPES = {"datetime64[ms]"}
    NUMERIC_TYPES = {"float32"}
    FLOAT_TYPES = {"float64"}
    INTEGER_TYPES = {"int16"}
    TIMEDELTA_TYPES = {"timedelta64[s]"}
    # To save time, we skip tests marked "pytest.mark.xfail"
    pytest_xfail = pytest.mark.skipif


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
def test_series_binop(request, arithmetic_op, obj_class):
    request.applymarker(
        pytest.mark.xfail(
            arithmetic_op is operator.floordiv,
            reason="https://github.com/rapidsai/cudf/issues/17073",
        )
    )
    nelem = 1000
    arr1 = utils.gen_rand("float64", nelem) * 10000
    # Keeping a low value because CUDA 'pow' has 2 full range error
    arr2 = utils.gen_rand("float64", nelem) * 10

    sr1 = Series(arr1)
    sr2 = Series(arr2)
    psr1 = sr1.to_pandas()
    psr2 = sr2.to_pandas()

    if obj_class == "Index":
        sr1 = Index(sr1)
        sr2 = Index(sr2)

    expect = arithmetic_op(psr1, psr2)
    result = arithmetic_op(sr1, sr2)

    if obj_class == "Index":
        result = Series(result)

    assert_eq(result, expect)


def test_series_binop_concurrent(arithmetic_op):
    def func(index):
        rng = np.random.default_rng(seed=0)
        arr = rng.random(100) * 10
        sr = Series(arr)

        result = arithmetic_op(sr.astype("int32"), sr)
        expect = arithmetic_op(arr.astype("int32"), arr)

        np.testing.assert_almost_equal(result.to_numpy(), expect, decimal=5)

    indices = range(10)
    with ThreadPoolExecutor(4) as e:  # four processes
        list(e.map(func, indices))


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
def test_series_binop_scalar(arithmetic_op, obj_class):
    nelem = 10
    rng = np.random.default_rng(seed=0)
    arr = rng.random(nelem)
    rhs = rng.choice(arr).item()

    sr = Series(arr)
    if obj_class == "Index":
        sr = Index(sr)

    result = arithmetic_op(sr, rhs)

    if obj_class == "Index":
        result = Series(result)

    np.testing.assert_almost_equal(result.to_numpy(), arithmetic_op(arr, rhs))


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("binop", [operator.and_, operator.or_, operator.xor])
def test_series_bitwise_binop(
    binop, obj_class, integer_types_as_str, integer_types_as_str2
):
    rng = np.random.default_rng(seed=0)
    arr1 = (rng.random(100) * 100).astype(integer_types_as_str)
    sr1 = Series(arr1)

    arr2 = (rng.random(100) * 100).astype(integer_types_as_str2)
    sr2 = Series(arr2)

    if obj_class == "Index":
        sr1 = Index(sr1)
        sr2 = Index(sr2)

    result = binop(sr1, sr2)

    if obj_class == "Index":
        result = Series(result)

    np.testing.assert_almost_equal(result.to_numpy(), binop(arr1, arr2))


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
def test_series_compare(
    comparison_op, obj_class, numeric_and_temporal_types_as_str
):
    rng = np.random.default_rng(seed=0)
    arr1 = rng.integers(0, 100, 100).astype(numeric_and_temporal_types_as_str)
    arr2 = rng.integers(0, 100, 100).astype(numeric_and_temporal_types_as_str)
    sr1 = Series(arr1)
    sr2 = Series(arr2)

    if obj_class == "Index":
        sr1 = Index(sr1)
        sr2 = Index(sr2)

    result1 = comparison_op(sr1, sr1)
    result2 = comparison_op(sr2, sr2)
    result3 = comparison_op(sr1, sr2)

    if obj_class == "Index":
        result1 = Series(result1)
        result2 = Series(result2)
        result3 = Series(result3)

    np.testing.assert_equal(result1.to_numpy(), comparison_op(arr1, arr1))
    np.testing.assert_equal(result2.to_numpy(), comparison_op(arr2, arr2))
    np.testing.assert_equal(result3.to_numpy(), comparison_op(arr1, arr2))


@pytest.mark.parametrize(
    "dtype,val",
    [("int8", 200), ("int32", 2**32), ("uint8", -128), ("uint64", -1)],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_series_compare_integer(dtype, val, comparison_op, reverse):
    # Tests that these actually work, even though they are out of bound.
    force_cast_val = np.array(val).astype(dtype)
    sr = Series(
        [np.iinfo(dtype).min, np.iinfo(dtype).max, force_cast_val, None],
        dtype=dtype,
    )
    # We expect the same result as comparing to a value within range (e.g. 0)
    # except that a NULL value evaluates to False
    exp = False
    if reverse:
        if comparison_op(val, 0):
            exp = True
        res = comparison_op(val, sr)
    else:
        if comparison_op(0, val):
            exp = True
        res = comparison_op(sr, val)

    expected = Series([exp, exp, exp, None])
    assert_eq(res, expected)


@pytest.mark.parametrize(
    "dtypes",
    [
        *itertools.combinations_with_replacement(DATETIME_TYPES, 2),
        *itertools.combinations_with_replacement(TIMEDELTA_TYPES, 2),
        *itertools.combinations_with_replacement(NUMERIC_TYPES, 2),
        *itertools.combinations_with_replacement(STRING_TYPES, 2),
    ],
)
def test_series_compare_nulls(comparison_op, dtypes):
    ltype, rtype = dtypes

    ldata = [1, 2, None, None, 5]
    rdata = [2, 1, None, 4, None]

    lser = Series(ldata, dtype=ltype)
    rser = Series(rdata, dtype=rtype)

    lmask = ~lser.isnull()
    rmask = ~rser.isnull()

    expect_mask = np.logical_and(lmask, rmask)
    expect = cudf.Series([None] * 5, dtype="bool")
    expect[expect_mask] = comparison_op(lser[expect_mask], rser[expect_mask])

    got = comparison_op(lser, rser)
    assert_eq(expect, got)


@pytest.fixture
def str_series_cmp_data():
    return pd.Series(["a", "b", None, "d", "e", None], dtype="string")


@pytest.fixture(ids=["eq", "ne"], params=[operator.eq, operator.ne])
def str_series_compare_num_cmpop(request):
    return request.param


@pytest.fixture(ids=["int", "float", "bool"], params=[1, 1.5, True])
def cmp_scalar(request):
    return request.param


def test_str_series_compare_str(str_series_cmp_data, comparison_op):
    expect = comparison_op(str_series_cmp_data, "a")
    got = comparison_op(Series.from_pandas(str_series_cmp_data), "a")

    assert_eq(expect, got.to_pandas(nullable=True))


def test_str_series_compare_str_reflected(str_series_cmp_data, comparison_op):
    expect = comparison_op("a", str_series_cmp_data)
    got = comparison_op("a", Series.from_pandas(str_series_cmp_data))

    assert_eq(expect, got.to_pandas(nullable=True))


def test_str_series_compare_num(
    str_series_cmp_data, str_series_compare_num_cmpop, cmp_scalar
):
    expect = str_series_compare_num_cmpop(str_series_cmp_data, cmp_scalar)
    got = str_series_compare_num_cmpop(
        Series.from_pandas(str_series_cmp_data), cmp_scalar
    )

    assert_eq(expect, got.to_pandas(nullable=True))


def test_str_series_compare_num_reflected(
    str_series_cmp_data, str_series_compare_num_cmpop, cmp_scalar
):
    expect = str_series_compare_num_cmpop(cmp_scalar, str_series_cmp_data)
    got = str_series_compare_num_cmpop(
        cmp_scalar, Series.from_pandas(str_series_cmp_data)
    )

    assert_eq(expect, got.to_pandas(nullable=True))


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("dtype", [*utils.NUMERIC_TYPES, "datetime64[ms]"])
def test_series_compare_scalar(comparison_op, obj_class, dtype):
    rng = np.random.default_rng(seed=0)
    arr1 = rng.integers(0, 100, 100).astype(dtype)
    sr1 = Series(arr1)
    rhs = rng.choice(arr1).item()

    if obj_class == "Index":
        sr1 = Index(sr1)

    result1 = comparison_op(sr1, rhs)
    result2 = comparison_op(rhs, sr1)

    if obj_class == "Index":
        result1 = Series(result1)
        result2 = Series(result2)

    np.testing.assert_equal(result1.to_numpy(), comparison_op(arr1, rhs))
    np.testing.assert_equal(result2.to_numpy(), comparison_op(rhs, arr1))


_nulls = ["none", "some"]


@pytest.mark.parametrize("lhs_nulls", _nulls)
@pytest.mark.parametrize("rhs_nulls", _nulls)
def test_validity_add(lhs_nulls, rhs_nulls):
    nelem = 10
    rng = np.random.default_rng(seed=0)
    # LHS
    lhs_data = rng.random(nelem)
    if lhs_nulls == "some":
        lhs_mask = utils.random_bitmask(nelem)
        lhs_bitmask = utils.expand_bits_to_bytes(lhs_mask)[:nelem]
        lhs_null_count = utils.count_zero(lhs_bitmask)
        assert lhs_null_count >= 0
        lhs = Series._from_column(as_column(lhs_data).set_mask(lhs_mask))
        assert lhs.null_count == lhs_null_count
    else:
        lhs = Series(lhs_data)
    # RHS
    rhs_data = rng.random(nelem)
    if rhs_nulls == "some":
        rhs_mask = utils.random_bitmask(nelem)
        rhs_bitmask = utils.expand_bits_to_bytes(rhs_mask)[:nelem]
        rhs_null_count = utils.count_zero(rhs_bitmask)
        assert rhs_null_count >= 0
        rhs = Series._from_column(as_column(rhs_data).set_mask(rhs_mask))
        assert rhs.null_count == rhs_null_count
    else:
        rhs = Series(rhs_data)
    # Result
    res = lhs + rhs
    if lhs_nulls == "some" and rhs_nulls == "some":
        res_mask = np.asarray(
            utils.expand_bits_to_bytes(lhs_mask & rhs_mask), dtype=np.bool_
        )[:nelem]
    if lhs_nulls == "some" and rhs_nulls == "none":
        res_mask = np.asarray(
            utils.expand_bits_to_bytes(lhs_mask), dtype=np.bool_
        )[:nelem]
    if lhs_nulls == "none" and rhs_nulls == "some":
        res_mask = np.asarray(
            utils.expand_bits_to_bytes(rhs_mask), dtype=np.bool_
        )[:nelem]
    # Fill NA values
    na_value = -10000
    got = res.fillna(na_value).to_numpy()
    expect = lhs_data + rhs_data
    if lhs_nulls == "some" or rhs_nulls == "some":
        expect[~res_mask] = na_value

    np.testing.assert_array_equal(expect, got)


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("binop", [operator.add, operator.mul])
def test_series_binop_mixed_dtype(
    binop, numeric_types_as_str, numeric_types_as_str2, obj_class
):
    nelem = 10
    rng = np.random.default_rng(seed=0)
    lhs = (rng.random(nelem) * nelem).astype(numeric_types_as_str)
    rhs = (rng.random(nelem) * nelem).astype(numeric_types_as_str2)

    sr1 = Series(lhs)
    sr2 = Series(rhs)

    if obj_class == "Index":
        sr1 = Index(sr1)
        sr2 = Index(sr2)

    result = binop(Series(sr1), Series(sr2))

    if obj_class == "Index":
        result = Series(result)

    np.testing.assert_almost_equal(result.to_numpy(), binop(lhs, rhs))


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
def test_series_cmpop_mixed_dtype(
    comparison_op, numeric_types_as_str, numeric_types_as_str2, obj_class
):
    nelem = 5
    rng = np.random.default_rng(seed=0)
    lhs = (rng.random(nelem) * nelem).astype(numeric_types_as_str)
    rhs = (rng.random(nelem) * nelem).astype(numeric_types_as_str2)

    sr1 = Series(lhs)
    sr2 = Series(rhs)

    if obj_class == "Index":
        sr1 = Index(sr1)
        sr2 = Index(sr2)

    result = comparison_op(Series(sr1), Series(sr2))

    if obj_class == "Index":
        result = Series(result)

    np.testing.assert_array_equal(result.to_numpy(), comparison_op(lhs, rhs))


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in power:RuntimeWarning"
)
@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in power:RuntimeWarning"
)
@pytest.mark.parametrize("obj_class", [cudf.Series, cudf.Index])
@pytest.mark.parametrize("scalar", [-1, 0, 1])
def test_series_reflected_ops_scalar(
    arithmetic_op, scalar, numeric_types_as_str, obj_class
):
    # create random series
    func = lambda x: arithmetic_op(scalar, x)  # noqa: E731
    random_series = utils.gen_rand(numeric_types_as_str, 100, low=10, seed=12)

    gs = obj_class(random_series)

    try:
        gs_result = func(gs)
    except OverflowError:
        # An error is fine, if pandas raises the same error:
        with pytest.raises(OverflowError):
            func(random_series)

        return

    # class typing
    if obj_class == "Index":
        gs = Series(gs)

    # pandas
    ps_result = func(random_series)

    # verify
    np.testing.assert_allclose(ps_result, gs_result.to_numpy())


def test_different_shapes_and_columns(request, arithmetic_op):
    if arithmetic_op is operator.pow:
        msg = "TODO: Support `pow(1, NaN) == 1` and `pow(NaN, 0) == 1`"
        request.applymarker(pytest.mark.xfail(reason=msg))

    # Empty frame on the right side
    pd_frame = arithmetic_op(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({}))
    cd_frame = arithmetic_op(cudf.DataFrame({"x": [1, 2]}), cudf.DataFrame({}))
    assert_eq(cd_frame, pd_frame)

    # Empty frame on the left side
    pd_frame = pd.DataFrame({}) + pd.DataFrame({"x": [1, 2]})
    cd_frame = cudf.DataFrame({}) + cudf.DataFrame({"x": [1, 2]})
    assert_eq(cd_frame, pd_frame)

    # Note: the below rely on a discrepancy between cudf and pandas
    # While pandas inserts columns in alphabetical order, cudf inserts in the
    # order of whichever column comes first. So the following code will not
    # work if the names of columns are reversed i.e. ('y', 'x') != ('x', 'y')

    # More rows on the left side
    pd_frame = pd.DataFrame({"x": [1, 2, 3]}) + pd.DataFrame({"y": [1, 2]})
    cd_frame = cudf.DataFrame({"x": [1, 2, 3]}) + cudf.DataFrame({"y": [1, 2]})
    assert_eq(cd_frame, pd_frame)

    # More rows on the right side
    pd_frame = pd.DataFrame({"x": [1, 2]}) + pd.DataFrame({"y": [1, 2, 3]})
    cd_frame = cudf.DataFrame({"x": [1, 2]}) + cudf.DataFrame({"y": [1, 2, 3]})
    assert_eq(cd_frame, pd_frame)


def test_different_shapes_and_same_columns(arithmetic_op):
    pd_frame = arithmetic_op(
        pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [1, 2, 3]})
    )
    cd_frame = arithmetic_op(
        cudf.DataFrame({"x": [1, 2]}), cudf.DataFrame({"x": [1, 2, 3]})
    )
    # cast x as float64 so it matches pandas dtype
    cd_frame["x"] = cd_frame["x"].astype(np.float64)
    assert_eq(cd_frame, pd_frame)


def test_different_shapes_and_columns_with_unaligned_indices(
    request, arithmetic_op
):
    if arithmetic_op is operator.pow:
        msg = "TODO: Support `pow(1, NaN) == 1` and `pow(NaN, 0) == 1`"
        request.applymarker(pytest.mark.xfail(reason=msg))

    # Test with a RangeIndex
    pdf1 = pd.DataFrame({"x": [4, 3, 2, 1], "y": [7, 3, 8, 6]})
    # Test with an Index
    pdf2 = pd.DataFrame(
        {"x": [1, 2, 3, 7], "y": [4, 5, 6, 7]}, index=[0, 1, 3, 4]
    )
    # Test with an Index in a different order
    pdf3 = pd.DataFrame(
        {"x": [4, 5, 6, 7], "y": [1, 2, 3, 7], "z": [0, 5, 3, 7]},
        index=[0, 3, 5, 3],
    )
    gdf1 = cudf.DataFrame.from_pandas(pdf1)
    gdf2 = cudf.DataFrame.from_pandas(pdf2)
    gdf3 = cudf.DataFrame.from_pandas(pdf3)

    pd_frame = arithmetic_op(arithmetic_op(pdf1, pdf2), pdf3)
    cd_frame = arithmetic_op(arithmetic_op(gdf1, gdf2), gdf3)
    # cast x and y as float64 so it matches pandas dtype
    cd_frame["x"] = cd_frame["x"].astype(np.float64)
    cd_frame["y"] = cd_frame["y"].astype(np.float64)

    # Sort both frames by index and then by all columns to ensure consistent ordering
    pd_sorted = pd_frame.sort_index().sort_values(list(pd_frame.columns))
    cd_sorted = cd_frame.sort_index().sort_values(list(cd_frame.columns))
    assert_eq(cd_sorted, pd_sorted)

    pdf1 = pd.DataFrame({"x": [1, 1]}, index=["a", "a"])
    pdf2 = pd.DataFrame({"x": [2]}, index=["a"])
    gdf1 = cudf.DataFrame.from_pandas(pdf1)
    gdf2 = cudf.DataFrame.from_pandas(pdf2)
    pd_frame = arithmetic_op(pdf1, pdf2)
    cd_frame = arithmetic_op(gdf1, gdf2)

    # Sort both frames consistently for comparison
    pd_sorted = pd_frame.sort_index().sort_values(list(pd_frame.columns))
    cd_sorted = cd_frame.sort_index().sort_values(list(cd_frame.columns))
    assert_eq(pd_sorted, cd_sorted)


@pytest.mark.parametrize(
    "pdf2",
    [
        pd.DataFrame({"a": [3, 2, 1]}, index=[3, 2, 1]),
        pd.DataFrame([3, 2]),
    ],
)
def test_df_different_index_shape(pdf2, comparison_op):
    df1 = cudf.DataFrame([1, 2, 3], index=[1, 2, 3])

    pdf1 = df1.to_pandas()
    df2 = cudf.DataFrame.from_pandas(pdf2)

    utils.assert_exceptions_equal(
        lfunc=comparison_op,
        rfunc=comparison_op,
        lfunc_args_and_kwargs=([pdf1, pdf2],),
        rfunc_args_and_kwargs=([df1, df2],),
    )


def test_boolean_scalar_binop(comparison_op):
    rng = np.random.default_rng(seed=0)
    psr = pd.Series(rng.choice([True, False], 10))
    gsr = cudf.from_pandas(psr)
    assert_eq(comparison_op(psr, True), comparison_op(gsr, True))
    assert_eq(comparison_op(psr, False), comparison_op(gsr, False))


@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("fill_value", [None, 27])
def test_operator_func_between_series(
    float_types_as_str, arithmetic_op_method, has_nulls, fill_value
):
    count = 1000
    gdf_series_a = utils.gen_rand_series(
        float_types_as_str, count, has_nulls=has_nulls, stride=10000
    )
    gdf_series_b = utils.gen_rand_series(
        float_types_as_str, count, has_nulls=has_nulls, stride=100
    )
    pdf_series_a = gdf_series_a.to_pandas()
    pdf_series_b = gdf_series_b.to_pandas()

    gdf_result = getattr(gdf_series_a, arithmetic_op_method)(
        gdf_series_b, fill_value=fill_value
    )
    pdf_result = getattr(pdf_series_a, arithmetic_op_method)(
        pdf_series_b, fill_value=fill_value
    )

    assert_eq(pdf_result, gdf_result)


@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("fill_value", [None, 27])
def test_operator_func_series_and_scalar(
    float_types_as_str, arithmetic_op_method, has_nulls, fill_value
):
    count = 1000
    scalar = 59
    gdf_series = utils.gen_rand_series(
        float_types_as_str, count, has_nulls=has_nulls, stride=10000
    )
    pdf_series = gdf_series.to_pandas()

    gdf_series_result = getattr(gdf_series, arithmetic_op_method)(
        scalar,
        fill_value=fill_value,
    )
    pdf_series_result = getattr(pdf_series, arithmetic_op_method)(
        scalar,
        fill_value=fill_value,
    )

    assert_eq(pdf_series_result, gdf_series_result)


@pytest.mark.parametrize("fill_value", [0, 1, None, np.nan])
@pytest.mark.parametrize("scalar_a", [0, 1, None, np.nan])
@pytest.mark.parametrize("scalar_b", [0, 1, None, np.nan])
def test_operator_func_between_series_logical(
    float_types_as_str, comparison_op_method, scalar_a, scalar_b, fill_value
):
    gdf_series_a = Series([scalar_a], nan_as_null=False).astype(
        float_types_as_str
    )
    gdf_series_b = Series([scalar_b], nan_as_null=False).astype(
        float_types_as_str
    )

    pdf_series_a = gdf_series_a.to_pandas(nullable=True)
    pdf_series_b = gdf_series_b.to_pandas(nullable=True)

    gdf_series_result = getattr(gdf_series_a, comparison_op_method)(
        gdf_series_b, fill_value=fill_value
    )
    pdf_series_result = getattr(pdf_series_a, comparison_op_method)(
        pdf_series_b, fill_value=fill_value
    )
    expect = pdf_series_result
    got = gdf_series_result.to_pandas(nullable=True)

    # If fill_value is np.nan, things break down a bit,
    # because setting a NaN into a pandas nullable float
    # array still gets transformed to <NA>. As such,
    # pd_series_with_nulls.fillna(np.nan) has no effect.
    if (
        (pdf_series_a.isnull().sum() != pdf_series_b.isnull().sum())
        and np.isscalar(fill_value)
        and np.isnan(fill_value)
    ):
        with pytest.raises(AssertionError):
            assert_eq(expect, got)
        return
    assert_eq(expect, got)


@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("scalar", [-59.0, np.nan, 0, 59.0])
@pytest.mark.parametrize("fill_value", [None, 1.0])
def test_operator_func_series_and_scalar_logical(
    request,
    float_types_as_str,
    comparison_op_method,
    has_nulls,
    scalar,
    fill_value,
):
    request.applymarker(
        pytest.mark.xfail(
            PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION
            and fill_value == 1.0
            and scalar is np.nan
            and (
                has_nulls
                or (not has_nulls and comparison_op_method not in {"eq", "ne"})
            ),
            reason="https://github.com/pandas-dev/pandas/issues/57447",
        )
    )
    if has_nulls:
        gdf_series = cudf.Series(
            [-1.0, 0, cudf.NA, 1.1], dtype=float_types_as_str
        )
    else:
        gdf_series = cudf.Series(
            [-1.0, 0, 10.5, 1.1], dtype=float_types_as_str
        )
    pdf_series = gdf_series.to_pandas(nullable=True)
    gdf_series_result = getattr(gdf_series, comparison_op_method)(
        scalar,
        fill_value=fill_value,
    )
    pdf_series_result = getattr(pdf_series, comparison_op_method)(
        scalar, fill_value=fill_value
    )

    expect = pdf_series_result
    got = gdf_series_result.to_pandas(nullable=True)

    assert_eq(expect, got)


@pytest.mark.parametrize("nulls", _nulls)
@pytest.mark.parametrize("fill_value", [None, 27])
@pytest.mark.parametrize("other", ["df", "scalar"])
def test_operator_func_dataframe(
    arithmetic_op_method, nulls, fill_value, other
):
    num_rows = 100
    num_cols = 3

    def gen_df():
        rng = np.random.default_rng(seed=0)
        pdf = pd.DataFrame()
        from string import ascii_lowercase

        cols = rng.choice(num_cols + 5, num_cols, replace=False)

        for i in range(num_cols):
            colname = ascii_lowercase[cols[i]]
            data = utils.gen_rand("float64", num_rows) * 10000
            if nulls == "some":
                idx = rng.choice(
                    num_rows, size=int(num_rows / 2), replace=False
                )
                data[idx] = np.nan
            pdf[colname] = data
        return pdf

    pdf1 = gen_df()
    pdf2 = gen_df() if other == "df" else 59.0
    gdf1 = cudf.DataFrame.from_pandas(pdf1)
    gdf2 = cudf.DataFrame.from_pandas(pdf2) if other == "df" else 59.0

    got = getattr(gdf1, arithmetic_op_method)(gdf2, fill_value=fill_value)
    expect = getattr(pdf1, arithmetic_op_method)(pdf2, fill_value=fill_value)[
        list(got._data)
    ]

    assert_eq(expect, got)


@pytest.mark.parametrize("nulls", _nulls)
@pytest.mark.parametrize("other", ["df", "scalar"])
def test_logical_operator_func_dataframe(comparison_op_method, nulls, other):
    num_rows = 100
    num_cols = 3

    def gen_df():
        rng = np.random.default_rng(seed=0)
        pdf = pd.DataFrame()
        from string import ascii_lowercase

        cols = rng.choice(num_cols + 5, num_cols, replace=False)

        for i in range(num_cols):
            colname = ascii_lowercase[cols[i]]
            data = utils.gen_rand("float64", num_rows) * 10000
            if nulls == "some":
                idx = rng.choice(
                    num_rows, size=int(num_rows / 2), replace=False
                )
                data[idx] = np.nan
            pdf[colname] = data
        return pdf

    pdf1 = gen_df()
    pdf2 = gen_df() if other == "df" else 59.0
    gdf1 = cudf.DataFrame.from_pandas(pdf1, nan_as_null=False)
    gdf2 = (
        cudf.DataFrame.from_pandas(pdf2, nan_as_null=False)
        if other == "df"
        else 59.0
    )

    got = getattr(gdf1, comparison_op_method)(gdf2)
    expect = getattr(pdf1, comparison_op_method)(pdf2)[list(got._data)]

    assert_eq(expect, got)


@pytest.mark.parametrize("rhs", [0, 1, 10])
def test_binop_bool_uint(request, binary_op_method, rhs):
    if binary_op_method in {"rmod", "rfloordiv"}:
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/12162"
            ),
        )
    psr = pd.Series([True, False, False])
    gsr = cudf.from_pandas(psr)
    assert_eq(
        getattr(psr, binary_op_method)(rhs),
        getattr(gsr, binary_op_method)(rhs),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "series_dtype", (np.int8, np.uint8, np.int64, np.uint64)
)
@pytest.mark.parametrize(
    "divisor_dtype",
    (
        np.int8,
        np.uint8,
        np.int64,
        np.uint64,
    ),
)
@pytest.mark.parametrize("scalar_divisor", [False, True])
def test_floordiv_zero_float64(series_dtype, divisor_dtype, scalar_divisor):
    sr = pd.Series([1, 2, 3], dtype=series_dtype)
    cr = cudf.from_pandas(sr)

    if scalar_divisor:
        pd_div = divisor_dtype(0)
        cudf_div = pd_div
    else:
        pd_div = pd.Series([0], dtype=divisor_dtype)
        cudf_div = cudf.from_pandas(pd_div)
    assert_eq(sr // pd_div, cr // cudf_div)


@pytest.mark.parametrize("scalar_divisor", [False, True])
@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/12162")
def test_floordiv_zero_bool(scalar_divisor):
    sr = pd.Series([True, True, False], dtype=np.bool_)
    cr = cudf.from_pandas(sr)

    if scalar_divisor:
        pd_div = np.bool_(0)
        cudf_div = pd_div
    else:
        pd_div = pd.Series([0], dtype=np.bool_)
        cudf_div = cudf.from_pandas(pd_div)

    with pytest.raises((NotImplementedError, ZeroDivisionError)):
        sr // pd_div
    with pytest.raises((NotImplementedError, ZeroDivisionError)):
        cr // cudf_div


def test_rmod_zero_nan(numeric_and_bool_types_as_str, request):
    request.applymarker(
        pytest.mark.xfail(
            numeric_and_bool_types_as_str == "bool",
            reason="pandas returns int8, cuDF returns int64",
        )
    )
    sr = pd.Series([1, 1, 0], dtype=numeric_and_bool_types_as_str)
    cr = cudf.from_pandas(sr)
    assert_eq(1 % sr, 1 % cr)
    expected_dtype = (
        np.float64 if cr.dtype.kind != "f" else numeric_and_bool_types_as_str
    )
    assert_eq(1 % cr, cudf.Series([0, 0, None], dtype=expected_dtype))


def test_series_misc_binop():
    pds = pd.Series([1, 2, 4], name="abc xyz")
    gds = cudf.Series([1, 2, 4], name="abc xyz")

    assert_eq(pds + 1, gds + 1)
    assert_eq(1 + pds, 1 + gds)

    assert_eq(pds + pds, gds + gds)

    pds1 = pd.Series([1, 2, 4], name="hello world")
    gds1 = cudf.Series([1, 2, 4], name="hello world")

    assert_eq(pds + pds1, gds + gds1)
    assert_eq(pds1 + pds, gds1 + gds)

    assert_eq(pds1 + pds + 5, gds1 + gds + 5)


def test_int8_float16_binop():
    a = cudf.Series([1], dtype="int8")
    b = np.float16(2)
    expect = cudf.Series([0.5])
    got = a / b
    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("dtype", ["int64", "float64", "str"])
def test_vector_to_none_binops(dtype):
    data = Series([1, 2, 3, None], dtype=dtype)

    expect = Series([None] * 4).astype(dtype)
    got = data + None

    assert_eq(expect, got)


def is_timezone_aware_dtype(dtype: str) -> bool:
    return bool(re.match(r"^datetime64\[ns, .+\]$", dtype))


@pytest.mark.parametrize("n_periods", [0, 1, -12])
@pytest.mark.parametrize(
    "frequency",
    [
        "months",
        "years",
        "days",
        "hours",
        "minutes",
        "seconds",
        "microseconds",
        "nanoseconds",
    ],
)
@pytest.mark.parametrize(
    "dtype, components",
    [
        ["datetime64[ns]", "00.012345678"],
        ["datetime64[us]", "00.012345"],
        ["datetime64[ms]", "00.012"],
        ["datetime64[s]", "00"],
        ["datetime64[ns, Asia/Kathmandu]", "00.012345678"],
    ],
)
@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_datetime_dateoffset_binaryop(
    request, n_periods, frequency, dtype, components, op
):
    request.applymarker(
        pytest.mark.xfail(
            PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION
            and dtype in {"datetime64[ms]", "datetime64[s]"}
            and frequency == "microseconds"
            and n_periods == 0,
            reason="https://github.com/pandas-dev/pandas/issues/57448",
        )
    )
    if (
        not PANDAS_GE_220
        and dtype in {"datetime64[ms]", "datetime64[s]"}
        and frequency in ("microseconds", "nanoseconds")
        and n_periods != 0
    ):
        pytest.skip(reason="https://github.com/pandas-dev/pandas/pull/55595")
    if (
        not PANDAS_GE_220
        and dtype == "datetime64[us]"
        and frequency == "nanoseconds"
        and n_periods != 0
    ):
        pytest.skip(reason="https://github.com/pandas-dev/pandas/pull/55595")

    date_col = [
        f"2000-01-01 00:00:{components}",
        f"2000-01-31 00:00:{components}",
        f"2000-02-29 00:00:{components}",
    ]
    if is_timezone_aware_dtype(dtype):
        # Construct naive datetime64[ns] Series
        gsr = cudf.Series(date_col, dtype="datetime64[ns]")
        psr = gsr.to_pandas()

        # Convert to timezone-aware (both cudf and pandas)
        gsr = gsr.dt.tz_localize("UTC").dt.tz_convert("Asia/Kathmandu")
        psr = psr.dt.tz_localize("UTC").dt.tz_convert("Asia/Kathmandu")
    else:
        gsr = cudf.Series(date_col, dtype=dtype)
        psr = gsr.to_pandas()

    kwargs = {frequency: n_periods}

    goffset = cudf.DateOffset(**kwargs)
    poffset = pd.DateOffset(**kwargs)

    expect = op(psr, poffset)
    got = op(gsr, goffset)

    if is_timezone_aware_dtype(dtype):
        assert isinstance(expect.dtype, pd.DatetimeTZDtype)
        assert str(expect.dtype.tz) == str(got.dtype.tz)
        expect = expect.dt.tz_convert("UTC")
        got = got.dt.tz_convert("UTC")

    assert_eq(expect, got)

    expect = op(psr, -poffset)
    got = op(gsr, -goffset)

    if is_timezone_aware_dtype(dtype):
        assert isinstance(expect.dtype, pd.DatetimeTZDtype)
        assert str(expect.dtype.tz) == str(got.dtype.tz)
        expect = expect.dt.tz_convert("UTC")
        got = got.dt.tz_convert("UTC")

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"months": 2, "years": 5},
        {"microseconds": 1, "seconds": 1},
        {"months": 2, "years": 5, "seconds": 923, "microseconds": 481},
        {"milliseconds": 4},
        {"milliseconds": 4, "years": 2},
        {"nanoseconds": 12},
    ],
)
@pytest.mark.filterwarnings(
    "ignore:Non-vectorized DateOffset:pandas.errors.PerformanceWarning"
)
@pytest.mark.filterwarnings(
    "ignore:Discarding nonzero nanoseconds:UserWarning"
)
@pytest.mark.parametrize("op", [operator.add, operator.sub])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_datetime_dateoffset_binaryop_multiple(request, kwargs, op):
    gsr = cudf.Series(
        [
            "2000-01-01 00:00:00.012345678",
            "2000-01-31 00:00:00.012345678",
            "2000-02-29 00:00:00.012345678",
        ],
        dtype="datetime64[ns]",
    )
    psr = gsr.to_pandas()

    poffset = pd.DateOffset(**kwargs)
    goffset = cudf.DateOffset(**kwargs)

    expect = op(psr, poffset)
    got = op(gsr, goffset)

    assert_eq(expect, got)


@pytest.mark.parametrize("n_periods", [0, 1, -12])
@pytest.mark.parametrize(
    "frequency",
    [
        "months",
        "years",
        "days",
        "hours",
        "minutes",
        "seconds",
        "microseconds",
        "nanoseconds",
    ],
)
@pytest.mark.parametrize(
    "dtype, components",
    [
        ["datetime64[ns]", "00.012345678"],
        ["datetime64[us]", "00.012345"],
        ["datetime64[ms]", "00.012"],
        ["datetime64[s]", "00"],
    ],
)
def test_datetime_dateoffset_binaryop_reflected(
    n_periods, frequency, dtype, components
):
    if (
        not PANDAS_GE_220
        and dtype in {"datetime64[ms]", "datetime64[s]"}
        and frequency in ("microseconds", "nanoseconds")
        and n_periods != 0
    ):
        pytest.skip(reason="https://github.com/pandas-dev/pandas/pull/55595")
    if (
        not PANDAS_GE_220
        and dtype == "datetime64[us]"
        and frequency == "nanoseconds"
        and n_periods != 0
    ):
        pytest.skip(reason="https://github.com/pandas-dev/pandas/pull/55595")

    date_col = [
        f"2000-01-01 00:00:{components}",
        f"2000-01-31 00:00:{components}",
        f"2000-02-29 00:00:{components}",
    ]
    gsr = cudf.Series(date_col, dtype=dtype)
    psr = gsr.to_pandas()  # converts to nanos

    kwargs = {frequency: n_periods}

    goffset = cudf.DateOffset(**kwargs)
    poffset = pd.DateOffset(**kwargs)

    expect = poffset + psr
    got = goffset + gsr

    # TODO: Remove check_dtype once we get some clarity on:
    # https://github.com/pandas-dev/pandas/issues/57448
    assert_eq(expect, got, check_dtype=False)

    with pytest.raises(TypeError):
        poffset - psr

    with pytest.raises(TypeError):
        goffset - gsr


@pytest.mark.parametrize("frame", [cudf.Series, cudf.Index, cudf.DataFrame])
@pytest.mark.parametrize(
    "dtype", ["int", "str", "datetime64[s]", "timedelta64[s]", "category"]
)
def test_binops_with_lhs_numpy_scalar(frame, dtype):
    data = [1, 2, 3, 4, 5]

    data = (
        frame({"a": data}, dtype=dtype)
        if isinstance(frame, cudf.DataFrame)
        else frame(data, dtype=dtype)
    )

    if dtype == "datetime64[s]":
        val = cudf.dtype(dtype).type(4, "s")
    elif dtype == "timedelta64[s]":
        val = cudf.dtype(dtype).type(4, "s")
    elif dtype == "category":
        val = np.int64(4)
    elif dtype == "str":
        val = str(4)
    else:
        val = cudf.dtype(dtype).type(4)

    # Compare equality with series on left side to dispatch to the pandas/cudf
    # __eq__ operator and avoid a DeprecationWarning from numpy.
    expected = data.to_pandas() == val
    got = data == val

    assert_eq(expected, got)


def test_binops_with_NA_consistent(
    numeric_and_temporal_types_as_str, comparison_op_method
):
    data = [1, 2, 3]
    sr = cudf.Series(data, dtype=numeric_and_temporal_types_as_str)

    result = getattr(sr, comparison_op_method)(cudf.NA)
    if sr.dtype.kind in "mM":
        assert result.null_count == len(data)
    else:
        if comparison_op_method == "ne":
            expect_all = True
        else:
            expect_all = False
        assert (result == expect_all).all()


@pytest.mark.parametrize(
    "op, lhs, l_dtype, rhs, r_dtype, expect, expect_dtype",
    [
        (
            operator.add,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["3.0", "4.0"],
            cudf.Decimal64Dtype(scale=2, precision=4),
        ),
        (
            operator.add,
            2,
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["3.5", "4.0"],
            cudf.Decimal64Dtype(scale=2, precision=4),
        ),
        (
            operator.add,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["3.75", "3.005"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=17),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["100.1", "200.2"],
            cudf.Decimal128Dtype(scale=3, precision=23),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=1, precision=2),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", "0.995"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=1, precision=2),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", "0.995"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=10),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=6, precision=10),
            ["99.9", "199.8"],
            cudf.Decimal128Dtype(scale=6, precision=19),
        ),
        (
            operator.sub,
            2,
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.25", "0.995"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.mul,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["1.5", "3.0"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["2.25", "6.0"],
            cudf.Decimal64Dtype(scale=5, precision=8),
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["10.0", "40.0"],
            cudf.Decimal64Dtype(scale=1, precision=8),
        ),
        (
            operator.mul,
            ["1000", "2000"],
            cudf.Decimal64Dtype(scale=-3, precision=4),
            ["0.343", "0.500"],
            cudf.Decimal64Dtype(scale=3, precision=3),
            ["343.0", "1000.0"],
            cudf.Decimal64Dtype(scale=0, precision=8),
        ),
        (
            operator.mul,
            200,
            cudf.Decimal64Dtype(scale=3, precision=6),
            ["0.343", "0.500"],
            cudf.Decimal64Dtype(scale=3, precision=6),
            ["68.60", "100.0"],
            cudf.Decimal64Dtype(scale=3, precision=10),
        ),
        (
            operator.truediv,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=4),
            ["1.5", "3.0"],
            cudf.Decimal64Dtype(scale=1, precision=4),
            ["1.0", "0.6"],
            cudf.Decimal64Dtype(scale=7, precision=10),
        ),
        (
            operator.truediv,
            ["110", "200"],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=2, precision=4),
            ["1000.0", "1000.0"],
            cudf.Decimal64Dtype(scale=6, precision=12),
        ),
        (
            operator.truediv,
            ["132.86", "15.25"],
            cudf.Decimal64Dtype(scale=4, precision=14),
            ["2.34", "8.50"],
            cudf.Decimal64Dtype(scale=2, precision=8),
            ["56.77", "1.79"],
            cudf.Decimal128Dtype(scale=13, precision=25),
        ),
        (
            operator.truediv,
            ["20", "20"],
            cudf.Decimal128Dtype(scale=2, precision=6),
            ["20", "20"],
            cudf.Decimal128Dtype(scale=2, precision=6),
            ["1.0", "1.0"],
            cudf.Decimal128Dtype(scale=9, precision=15),
        ),
        (
            operator.add,
            ["1.5", None, "2.0"],
            cudf.Decimal64Dtype(scale=1, precision=2),
            ["1.5", None, "2.0"],
            cudf.Decimal64Dtype(scale=1, precision=2),
            ["3.0", None, "4.0"],
            cudf.Decimal64Dtype(scale=1, precision=3),
        ),
        (
            operator.add,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["3.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.mul,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["1.5", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=5, precision=8),
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=10),
            ["0.1", None],
            cudf.Decimal64Dtype(scale=3, precision=12),
            ["10.0", None],
            cudf.Decimal128Dtype(scale=1, precision=23),
        ),
        (
            operator.eq,
            ["0.18", "0.42"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.18", "0.21"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [True, False],
            bool,
        ),
        (
            operator.eq,
            ["0.18", "0.42"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1800", "0.2100"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [True, False],
            bool,
        ),
        (
            operator.eq,
            ["100", None],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [True, None],
            bool,
        ),
        (
            operator.ne,
            ["0.06", "0.42"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.18", "0.42"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [True, False],
            bool,
        ),
        (
            operator.ne,
            ["1.33", "1.21"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1899", "1.21"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [True, False],
            bool,
        ),
        (
            operator.ne,
            ["300", None],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["110", "5500"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [True, None],
            bool,
        ),
        (
            operator.lt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [False, True, False],
            bool,
        ),
        (
            operator.lt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [False, True, False],
            bool,
        ),
        (
            operator.lt,
            ["200", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [False, None, False],
            bool,
        ),
        (
            operator.gt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [True, False, False],
            bool,
        ),
        (
            operator.gt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [True, False, False],
            bool,
        ),
        (
            operator.gt,
            ["300", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [True, None, False],
            bool,
        ),
        (
            operator.le,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [False, True, True],
            bool,
        ),
        (
            operator.le,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [False, True, True],
            bool,
        ),
        (
            operator.le,
            ["300", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [False, None, True],
            bool,
        ),
        (
            operator.ge,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [True, False, True],
            bool,
        ),
        (
            operator.ge,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [True, False, True],
            bool,
        ),
        (
            operator.ge,
            ["300", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [True, None, True],
            bool,
        ),
    ],
)
def test_binops_decimal(op, lhs, l_dtype, rhs, r_dtype, expect, expect_dtype):
    if isinstance(lhs, (int, float)):
        a = lhs
    else:
        a = utils._decimal_series(lhs, l_dtype)
    b = utils._decimal_series(rhs, r_dtype)
    expect = (
        utils._decimal_series(expect, expect_dtype)
        if isinstance(
            expect_dtype,
            (cudf.Decimal64Dtype, cudf.Decimal32Dtype, cudf.Decimal128Dtype),
        )
        else cudf.Series(expect, dtype=expect_dtype)
    )

    got = op(a, b)
    assert expect.dtype == got.dtype
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "op,lhs,l_dtype,rhs,r_dtype,expect,expect_dtype",
    [
        (
            "radd",
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["3.0", "4.0"],
            cudf.Decimal64Dtype(scale=2, precision=4),
        ),
        (
            "rsub",
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=10),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=6, precision=10),
            ["-99.9", "-199.8"],
            cudf.Decimal128Dtype(scale=6, precision=19),
        ),
        (
            "rmul",
            ["1000", "2000"],
            cudf.Decimal64Dtype(scale=-3, precision=4),
            ["0.343", "0.500"],
            cudf.Decimal64Dtype(scale=3, precision=3),
            ["343.0", "1000.0"],
            cudf.Decimal64Dtype(scale=0, precision=8),
        ),
        (
            "rtruediv",
            ["1.5", "0.5"],
            cudf.Decimal64Dtype(scale=3, precision=6),
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=3, precision=6),
            ["1.0", "4.0"],
            cudf.Decimal64Dtype(scale=10, precision=16),
        ),
    ],
)
def test_binops_reflect_decimal(
    op, lhs, l_dtype, rhs, r_dtype, expect, expect_dtype
):
    a = utils._decimal_series(lhs, l_dtype)
    b = utils._decimal_series(rhs, r_dtype)
    expect = utils._decimal_series(expect, expect_dtype)

    got = getattr(a, op)(b)
    assert expect.dtype == got.dtype
    assert_eq(expect, got)


@pytest.mark.parametrize("powers", [0, 1, 2])
def test_binops_decimal_pow(powers):
    s = cudf.Series(
        [
            decimal.Decimal("1.324324"),
            None,
            decimal.Decimal("2"),
            decimal.Decimal("3"),
            decimal.Decimal("5"),
        ]
    )
    ps = s.to_pandas()

    assert_eq(s**powers, ps**powers, check_dtype=False)


def test_binops_raise_error():
    s = cudf.Series([decimal.Decimal("1.324324")])

    with pytest.raises(TypeError):
        s // 1


@pytest.mark.parametrize(
    "op, ldata, ldtype, rdata, expected1, expected2",
    [
        (
            operator.eq,
            ["100", "41", None],
            cudf.Decimal64Dtype(scale=0, precision=5),
            [100, 42, 12],
            [True, False, None],
            [True, False, None],
        ),
        (
            operator.eq,
            ["100.000", "42.001", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 12],
            [True, False, None],
            [True, False, None],
        ),
        (
            operator.eq,
            ["100", "40", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 12],
            [True, False, None],
            [True, False, None],
        ),
        (
            operator.ne,
            ["100", "42", "24", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 40, 24, 12],
            [False, True, False, None],
            [False, True, False, None],
        ),
        (
            operator.ne,
            ["10.1", "88", "11", None],
            cudf.Decimal64Dtype(scale=1, precision=3),
            [10, 42, 11, 12],
            [True, True, False, None],
            [True, True, False, None],
        ),
        (
            operator.ne,
            ["100.000", "42", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            [False, False, True, None],
            [False, False, True, None],
        ),
        (
            operator.lt,
            ["100", "40", "28", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 42, 24, 12],
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.lt,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            [False, False, True, None],
            [False, True, False, None],
        ),
        (
            operator.lt,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.gt,
            ["100", "42", "20", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 40, 24, 12],
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.gt,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.gt,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            [False, False, True, None],
            [False, True, False, None],
        ),
        (
            operator.le,
            ["100", "40", "28", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 42, 24, 12],
            [True, True, False, None],
            [True, False, True, None],
        ),
        (
            operator.le,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            [True, False, True, None],
            [True, True, False, None],
        ),
        (
            operator.le,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            [True, True, False, None],
            [True, False, True, None],
        ),
        (
            operator.ge,
            ["100", "42", "20", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 40, 24, 12],
            [True, True, False, None],
            [True, False, True, None],
        ),
        (
            operator.ge,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            [True, True, False, None],
            [True, False, True, None],
        ),
        (
            operator.ge,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            [True, False, True, None],
            [True, True, False, None],
        ),
    ],
)
@pytest.mark.parametrize("reflected", [True, False])
def test_binops_decimal_comp_mixed_integer(
    op,
    ldata,
    ldtype,
    rdata,
    expected1,
    expected2,
    integer_types_as_str,
    reflected,
):
    """
    Tested compare operations:
        eq, lt, gt, le, ge
    Each operation has 3 decimal data setups, with scale from {==0, >0, <0}.
    Decimal precisions are sufficient to hold the digits.
    For each decimal data setup, there is at least one row that lead to one
    of the following compare results: {True, False, None}.
    """
    if not reflected:
        expected = cudf.Series(expected1, dtype=bool)
    else:
        expected = cudf.Series(expected2, dtype=bool)

    lhs = utils._decimal_series(ldata, ldtype)
    rhs = cudf.Series(rdata, dtype=integer_types_as_str)

    if reflected:
        rhs, lhs = lhs, rhs

    actual = op(lhs, rhs)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "op, lhs, l_dtype, rhs, expect, expect_dtype, reflect",
    [
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(1),
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["101.5", "201.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            False,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(1),
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["101.5", "201.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            True,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            False,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["200", "400"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            False,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["150", "300"],
            cudf.Decimal64Dtype(scale=-1, precision=6),
            False,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            True,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["200", "400"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            True,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["150", "300"],
            cudf.Decimal64Dtype(scale=-1, precision=6),
            True,
        ),
        (
            operator.truediv,
            ["1000", "2000"],
            cudf.Decimal64Dtype(scale=-2, precision=4),
            1,
            ["1000", "2000"],
            cudf.Decimal64Dtype(scale=6, precision=12),
            False,
        ),
        (
            operator.truediv,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=2, precision=5),
            decimal.Decimal(2),
            ["50", "100"],
            cudf.Decimal64Dtype(scale=6, precision=9),
            False,
        ),
        (
            operator.truediv,
            ["35.23", "54.91"],
            cudf.Decimal64Dtype(scale=2, precision=4),
            decimal.Decimal("1.5"),
            ["23.4", "36.6"],
            cudf.Decimal64Dtype(scale=6, precision=9),
            False,
        ),
        (
            operator.truediv,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=2, precision=5),
            1,
            ["0", "0"],
            cudf.Decimal64Dtype(scale=6, precision=9),
            True,
        ),
        (
            operator.truediv,
            ["1.2", "0.5"],
            cudf.Decimal64Dtype(scale=1, precision=6),
            decimal.Decimal(20),
            ["10", "40"],
            cudf.Decimal64Dtype(scale=7, precision=10),
            True,
        ),
        (
            operator.truediv,
            ["1.22", "5.24"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            decimal.Decimal("8.55"),
            ["7", "1"],
            cudf.Decimal64Dtype(scale=6, precision=9),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["98", "198"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("2.5"),
            ["97.5", "197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            False,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            4,
            ["96", "196"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["-98", "-198"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            4,
            ["-96", "-196"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("2.5"),
            ["-97.5", "-197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("2.5"),
            ["-97.5", "-197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            True,
        ),
    ],
)
def test_binops_decimal_scalar(
    op, lhs, l_dtype, rhs, expect, expect_dtype, reflect
):
    lhs = cudf.Series(
        [x if x is None else decimal.Decimal(x) for x in lhs],
        dtype=l_dtype,
    )
    expect = cudf.Series(
        [x if x is None else decimal.Decimal(x) for x in expect],
        dtype=expect_dtype,
    )

    if reflect:
        lhs, rhs = rhs, lhs

    got = op(lhs, rhs)
    assert expect.dtype == got.dtype
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "op, ldata, ldtype, rdata, expected1, expected2",
    [
        (
            operator.eq,
            ["100.00", "41", None],
            cudf.Decimal64Dtype(scale=0, precision=5),
            100,
            [True, False, None],
            [True, False, None],
        ),
        (
            operator.eq,
            ["100.123", "41", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [True, False, None],
            [True, False, None],
        ),
        (
            operator.ne,
            ["100.00", "41", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            [False, True, None],
            [False, True, None],
        ),
        (
            operator.ne,
            ["100.123", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [False, True, None],
            [False, True, None],
        ),
        (
            operator.gt,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            [False, False, True, None],
            [False, True, False, None],
        ),
        (
            operator.gt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [False, False, True, None],
            [False, True, False, None],
        ),
        (
            operator.ge,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            [True, False, True, None],
            [True, True, False, None],
        ),
        (
            operator.ge,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [True, False, True, None],
            [True, True, False, None],
        ),
        (
            operator.lt,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.lt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.le,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            [True, True, False, None],
            [True, False, True, None],
        ),
        (
            operator.le,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [True, True, False, None],
            [True, False, True, None],
        ),
    ],
)
@pytest.mark.parametrize("reflected", [True, False])
def test_binops_decimal_scalar_compare(
    op, ldata, ldtype, rdata, expected1, expected2, reflected
):
    """
    Tested compare operations:
        eq, lt, gt, le, ge
    Each operation has 3 data setups: pyints, Decimal, and
    For each data setup, there is at least one row that lead to one of the
    following compare results: {True, False, None}.
    """
    if not reflected:
        expected = cudf.Series(expected1, dtype=bool)
    else:
        expected = cudf.Series(expected2, dtype=bool)

    lhs = utils._decimal_series(ldata, ldtype)
    rhs = rdata

    if reflected:
        rhs, lhs = lhs, rhs

    actual = op(lhs, rhs)

    assert_eq(expected, actual)


@pytest.mark.parametrize("null_scalar", [None, cudf.NA, np.datetime64("NaT")])
def test_column_null_scalar_comparison(
    request, all_supported_types_as_str, null_scalar, comparison_op
):
    # This test is meant to validate that comparing
    # a series of any dtype with a null scalar produces
    # a new series where all the elements are <NA>.
    request.applymarker(
        pytest.mark.xfail(
            all_supported_types_as_str == "category",
            raises=ValueError,
            reason="Value ... not found in column",
        )
    )
    dtype = cudf.dtype(all_supported_types_as_str)

    if isinstance(null_scalar, np.datetime64):
        if dtype.kind not in "mM":
            pytest.skip(f"{null_scalar} not applicable for {dtype}")
        null_scalar = null_scalar.astype(dtype)

    data = [1, 2, 3, 4, 5]
    sr = cudf.Series(data, dtype=dtype)
    result = comparison_op(sr, null_scalar)

    assert result.isnull().all()


def test_equality_ops_index_mismatch(comparison_op_method):
    a = cudf.Series(
        [1, 2, 3, None, None, 4], index=["a", "b", "c", "d", "e", "f"]
    )
    b = cudf.Series(
        [-5, 4, 3, 2, 1, 0, 19, 11],
        index=["aa", "b", "c", "d", "e", "f", "y", "z"],
    )

    pa = a.to_pandas(nullable=True)
    pb = b.to_pandas(nullable=True)
    expected = getattr(pa, comparison_op_method)(pb)
    actual = getattr(a, comparison_op_method)(b).to_pandas(nullable=True)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "dtype",
    sorted(
        itertools.chain(
            NUMERIC_TYPES,
            DATETIME_TYPES,
            TIMEDELTA_TYPES,
            STRING_TYPES,
            ["category"],
        )
    ),
)
@pytest.mark.parametrize("null_case", ["neither", "left", "right", "both"])
def test_null_equals_columnops(dtype, null_case):
    # Generate tuples of:
    # (left_data, right_data, compare_bool
    # where compare_bool is the correct answer to
    # if the columns should compare as null equals

    def set_null_cases(column_l, column_r, case):
        if case == "neither":
            return column_l, column_r
        elif case == "left":
            column_l[1] = None
        elif case == "right":
            column_r[1] = None
        elif case == "both":
            column_l[1] = None
            column_r[1] = None
        else:
            raise ValueError("Unknown null case")
        return column_l, column_r

    data = [1, 2, 3]

    left = cudf.Series(data, dtype=dtype)
    right = cudf.Series(data, dtype=dtype)
    if null_case in {"left", "right"}:
        answer = False
    else:
        answer = True
    left, right = set_null_cases(left, right, null_case)
    assert left._column.equals(right._column) is answer


def test_add_series_to_dataframe():
    """Verify that missing columns result in NaNs, not NULLs."""
    assert cp.all(
        cp.isnan(
            (
                cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
                + cudf.Series([1, 2, 3], index=["a", "b", "c"])
            )["c"]
        )
    )


@pytest.mark.parametrize("obj_class", [cudf.Series, cudf.Index])
def test_binops_cupy_array(obj_class, arithmetic_op):
    # Skip 0 to not deal with NaNs from division.
    data = range(1, 100)
    lhs = obj_class(data)
    rhs = cp.array(data)
    assert (arithmetic_op(lhs, rhs) == arithmetic_op(lhs, lhs)).all()


@pytest.mark.parametrize("data", [None, [-9, 7], [12, 18]])
@pytest.mark.parametrize("scalar", [1, 3, 12, np.nan])
def test_empty_column(binary_op, data, scalar):
    gdf = cudf.DataFrame(columns=["a", "b"])
    if data is not None:
        gdf["a"] = data

    pdf = gdf.to_pandas()

    got = binary_op(gdf, scalar)
    expected = binary_op(pdf, scalar)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "df",
    [
        lambda: cudf.DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8], [10, 11, 12, 13], [14, 15, 16, 17]]
        ),
        pytest.param(
            lambda: cudf.DataFrame([[1, None, None, 4], [5, 6, 7, None]]),
            marks=pytest_xfail(
                reason="Cannot access Frame.values if frame contains nulls"
            ),
        ),
        lambda: cudf.DataFrame(
            [
                [1.2, 2.3, 3.4, 4.5],
                [5.6, 6.7, 7.8, 8.9],
                [7.43, 4.2, 23.2, 23.2],
                [9.1, 2.4, 4.5, 65.34],
            ]
        ),
        lambda: cudf.Series([14, 15, 16, 17]),
        lambda: cudf.Series([14.15, 15.16, 16.17, 17.18]),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        lambda: cudf.DataFrame([[9, 10], [11, 12], [13, 14], [15, 16]]),
        lambda: cudf.DataFrame(
            [[9.4, 10.5], [11.6, 12.7], [13.8, 14.9], [15.1, 16.2]]
        ),
        lambda: cudf.Series([5, 6, 7, 8]),
        lambda: cudf.Series([5.6, 6.7, 7.8, 8.9]),
        lambda: np.array([5, 6, 7, 8]),
        lambda: [25.5, 26.6, 27.7, 28.8],
    ],
)
def test_binops_dot(df, other):
    df = df()
    other = other()
    pdf = df.to_pandas()
    host_other = other.to_pandas() if hasattr(other, "to_pandas") else other

    expected = pdf @ host_other
    got = df @ other

    assert_eq(expected, got)


def test_binop_dot_preserve_index():
    ser = cudf.Series(range(2), index=["A", "B"])
    df = cudf.DataFrame(np.eye(2), columns=["A", "B"], index=["A", "B"])
    result = ser @ df
    expected = ser.to_pandas() @ df.to_pandas()
    assert_eq(result, expected)


def test_binop_series_with_repeated_index():
    # GH: #11094
    psr1 = pd.Series([1, 1], index=["a", "a"])
    psr2 = pd.Series([1], index=["a"])
    gsr1 = cudf.from_pandas(psr1)
    gsr2 = cudf.from_pandas(psr2)
    expected = psr1 - psr2
    got = gsr1 - gsr2
    assert_eq(expected, got)


def test_binop_integer_power_series_series():
    # GH: #10178
    gs_base = cudf.Series([3, -3, 8, -8])
    gs_exponent = cudf.Series([1, 1, 7, 7])
    ps_base = gs_base.to_pandas()
    ps_exponent = gs_exponent.to_pandas()
    expected = ps_base**ps_exponent
    got = gs_base**gs_exponent
    assert_eq(expected, got)


def test_binop_integer_power_series_int():
    # GH: #10178
    gs_base = cudf.Series([3, -3, 8, -8])
    exponent = 1
    ps_base = gs_base.to_pandas()
    expected = ps_base**exponent
    got = gs_base**exponent
    assert_eq(expected, got)


def test_binop_integer_power_int_series():
    # GH: #10178
    base = 3
    gs_exponent = cudf.Series([1, 1, 7, 7])
    ps_exponent = gs_exponent.to_pandas()
    expected = base**ps_exponent
    got = base**gs_exponent
    assert_eq(expected, got)


def test_binop_index_series(arithmetic_op):
    gi = cudf.Index([10, 11, 12])
    gs = cudf.Series([1, 2, 3])

    actual = arithmetic_op(gi, gs)
    expected = arithmetic_op(gi.to_pandas(), gs.to_pandas())

    assert_eq(expected, actual)


@pytest.mark.parametrize("name1", utils.SERIES_OR_INDEX_NAMES)
@pytest.mark.parametrize("name2", utils.SERIES_OR_INDEX_NAMES)
def test_binop_index_dt_td_series_with_names(name1, name2):
    gi = cudf.Index([1, 2, 3], dtype="datetime64[ns]", name=name1)
    gs = cudf.Series([10, 11, 12], dtype="timedelta64[ns]", name=name2)
    with warnings.catch_warnings():
        # Numpy raises a deprecation warning:
        # "elementwise comparison failed; this will raise an error "
        warnings.simplefilter("ignore", (DeprecationWarning,))

        expected = gi.to_pandas() + gs.to_pandas()
    actual = gi + gs

    assert_eq(expected, actual)


@pytest.mark.parametrize("data1", [[1, 2, 3], [10, 11, None]])
@pytest.mark.parametrize("data2", [[1, 2, 3], [10, 11, None]])
def test_binop_eq_ne_index_series(data1, data2):
    gi = cudf.Index(data1, dtype="datetime64[ns]", name=np.nan)
    gs = cudf.Series(data2, dtype="timedelta64[ns]", name="abc")

    actual = gi == gs
    expected = gi.to_pandas() == gs.to_pandas()

    assert_eq(expected, actual)

    actual = gi != gs
    expected = gi.to_pandas() != gs.to_pandas()

    assert_eq(expected, actual)


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
def test_binop_lhs_numpy_datetimelike_scalar(scalar):
    slr1 = scalar(1, "ms")
    slr2 = scalar(1, "ns")
    result = slr1 < cudf.Series([slr2])
    expected = slr1 < pd.Series([slr2])
    assert_eq(result, expected)

    result = slr2 < cudf.Series([slr1])
    expected = slr2 < pd.Series([slr1])
    assert_eq(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize(
    "data_left, data_right",
    [
        [[1, 2], [1, 2]],
        [[1, 2], [1, 3]],
    ],
)
def test_cat_non_cat_compare_ops(
    comparison_op, data_left, data_right, ordered
):
    pd_non_cat = pd.Series(data_left)
    pd_cat = pd.Series(
        data_right,
        dtype=pd.CategoricalDtype(categories=data_right, ordered=ordered),
    )

    cudf_non_cat = cudf.Series.from_pandas(pd_non_cat)
    cudf_cat = cudf.Series.from_pandas(pd_cat)

    if (
        not ordered and comparison_op not in {operator.eq, operator.ne}
    ) or comparison_op in {
        operator.gt,
        operator.lt,
        operator.le,
        operator.ge,
    }:
        with pytest.raises(TypeError):
            comparison_op(pd_non_cat, pd_cat)
        with pytest.raises(TypeError):
            comparison_op(cudf_non_cat, cudf_cat)
    else:
        expected = comparison_op(pd_non_cat, pd_cat)
        result = comparison_op(cudf_non_cat, cudf_cat)
        assert_eq(result, expected)


@pytest.mark.parametrize(
    "left_data, right_data",
    [[["a", "b"], [1, 2]], [[[1, 2, 3], [4, 5]], [{"a": 1}, {"a": 2}]]],
)
@pytest.mark.parametrize(
    "op, expected_data",
    [[operator.eq, [False, False]], [operator.ne, [True, True]]],
)
@pytest.mark.parametrize("with_na", [True, False])
def test_eq_ne_non_comparable_types(
    left_data, right_data, op, expected_data, with_na
):
    if with_na:
        left_data[0] = None
    left = cudf.Series(left_data)
    right = cudf.Series(right_data)
    result = op(left, right)
    if with_na:
        expected_data[0] = None
    expected = cudf.Series(expected_data)
    assert_eq(result, expected)


def test_binops_compare_stdlib_date_scalar(comparison_op):
    dt = datetime.date(2020, 1, 1)
    data = [dt]
    result = comparison_op(cudf.Series(data), dt)
    expected = comparison_op(pd.Series(data), dt)
    assert_eq(result, expected)

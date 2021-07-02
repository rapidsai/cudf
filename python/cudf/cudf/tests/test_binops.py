# Copyright (c) 2018-2021, NVIDIA CORPORATION.

from __future__ import division

import decimal
import operator
import random
from itertools import product

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core import Series
from cudf.core.index import as_index
from cudf.testing import _utils as utils
from cudf.utils.dtypes import (
    BOOL_TYPES,
    DATETIME_TYPES,
    FLOAT_TYPES,
    INTEGER_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
)

STRING_TYPES = {"str"}

_binops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.pow,
]


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("binop", _binops)
def test_series_binop(binop, obj_class):
    nelem = 1000
    arr1 = utils.gen_rand("float64", nelem) * 10000
    # Keeping a low value because CUDA 'pow' has 2 full range error
    arr2 = utils.gen_rand("float64", nelem) * 10

    sr1 = Series(arr1)
    sr2 = Series(arr2)

    if obj_class == "Index":
        sr1 = as_index(sr1)
        sr2 = as_index(sr2)

    result = binop(sr1, sr2)
    expect = binop(pd.Series(arr1), pd.Series(arr2))

    if obj_class == "Index":
        result = Series(result)

    utils.assert_eq(result, expect)


@pytest.mark.parametrize("binop", _binops)
def test_series_binop_concurrent(binop):
    def func(index):
        arr = np.random.random(100) * 10
        sr = Series(arr)

        result = binop(sr.astype("int32"), sr)
        expect = binop(arr.astype("int32"), arr)

        np.testing.assert_almost_equal(result.to_array(), expect, decimal=5)

    from concurrent.futures import ThreadPoolExecutor

    indices = range(10)
    with ThreadPoolExecutor(4) as e:  # four processes
        list(e.map(func, indices))


@pytest.mark.parametrize("use_cudf_scalar", [False, True])
@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("nelem,binop", list(product([1, 2, 100], _binops)))
def test_series_binop_scalar(nelem, binop, obj_class, use_cudf_scalar):
    arr = np.random.random(nelem)
    rhs = random.choice(arr).item()

    sr = Series(arr)
    if obj_class == "Index":
        sr = as_index(sr)

    if use_cudf_scalar:
        result = binop(sr, rhs)
    else:
        result = binop(sr, cudf.Scalar(rhs))

    if obj_class == "Index":
        result = Series(result)

    np.testing.assert_almost_equal(result.to_array(), binop(arr, rhs))


_bitwise_binops = [operator.and_, operator.or_, operator.xor]


_int_types = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
]


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("binop", _bitwise_binops)
@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype", list(product(_int_types, _int_types))
)
def test_series_bitwise_binop(binop, obj_class, lhs_dtype, rhs_dtype):
    arr1 = (np.random.random(100) * 100).astype(lhs_dtype)
    sr1 = Series(arr1)

    arr2 = (np.random.random(100) * 100).astype(rhs_dtype)
    sr2 = Series(arr2)

    if obj_class == "Index":
        sr1 = as_index(sr1)
        sr2 = as_index(sr2)

    result = binop(sr1, sr2)

    if obj_class == "Index":
        result = Series(result)

    np.testing.assert_almost_equal(result.to_array(), binop(arr1, arr2))


_logical_binops = [
    (operator.and_, operator.and_),
    (operator.or_, operator.or_),
    (np.logical_and, cudf.logical_and),
    (np.logical_or, cudf.logical_or),
]


@pytest.mark.parametrize("lhstype", _int_types + [np.bool_])
@pytest.mark.parametrize("rhstype", _int_types + [np.bool_])
@pytest.mark.parametrize("binop,cubinop", _logical_binops)
def test_series_logical_binop(lhstype, rhstype, binop, cubinop):
    arr1 = pd.Series(np.random.choice([True, False], 10))
    if lhstype is not np.bool_:
        arr1 = arr1 * (np.random.random(10) * 100).astype(lhstype)
    sr1 = Series(arr1)

    arr2 = pd.Series(np.random.choice([True, False], 10))
    if rhstype is not np.bool_:
        arr2 = arr2 * (np.random.random(10) * 100).astype(rhstype)
    sr2 = Series(arr2)

    result = cubinop(sr1, sr2)
    expect = binop(arr1, arr2)

    utils.assert_eq(result, expect)


_cmpops = [
    operator.lt,
    operator.gt,
    operator.le,
    operator.ge,
    operator.eq,
    operator.ne,
]


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("cmpop", _cmpops)
@pytest.mark.parametrize(
    "dtype", ["int8", "int32", "int64", "float32", "float64", "datetime64[ms]"]
)
def test_series_compare(cmpop, obj_class, dtype):
    arr1 = np.random.randint(0, 100, 100).astype(dtype)
    arr2 = np.random.randint(0, 100, 100).astype(dtype)
    sr1 = Series(arr1)
    sr2 = Series(arr2)

    if obj_class == "Index":
        sr1 = as_index(sr1)
        sr2 = as_index(sr2)

    result1 = cmpop(sr1, sr1)
    result2 = cmpop(sr2, sr2)
    result3 = cmpop(sr1, sr2)

    if obj_class == "Index":
        result1 = Series(result1)
        result2 = Series(result2)
        result3 = Series(result3)

    np.testing.assert_equal(result1.to_array(), cmpop(arr1, arr1))
    np.testing.assert_equal(result2.to_array(), cmpop(arr2, arr2))
    np.testing.assert_equal(result3.to_array(), cmpop(arr1, arr2))


def _series_compare_nulls_typegen():
    tests = []
    tests += list(product(DATETIME_TYPES, DATETIME_TYPES))
    tests += list(product(TIMEDELTA_TYPES, TIMEDELTA_TYPES))
    tests += list(product(NUMERIC_TYPES, NUMERIC_TYPES))
    tests += list(product(STRING_TYPES, STRING_TYPES))

    return tests


@pytest.mark.parametrize("cmpop", _cmpops)
@pytest.mark.parametrize("dtypes", _series_compare_nulls_typegen())
def test_series_compare_nulls(cmpop, dtypes):
    ltype, rtype = dtypes

    ldata = [1, 2, None, None, 5]
    rdata = [2, 1, None, 4, None]

    lser = Series(ldata, dtype=ltype)
    rser = Series(rdata, dtype=rtype)

    lmask = ~lser.isnull()
    rmask = ~rser.isnull()

    expect_mask = np.logical_and(lmask, rmask)
    expect = cudf.Series([None] * 5, dtype="bool")
    expect[expect_mask] = cmpop(lser[expect_mask], rser[expect_mask])

    got = cmpop(lser, rser)
    utils.assert_eq(expect, got)


@pytest.mark.parametrize(
    "obj", [pd.Series(["a", "b", None, "d", "e", None], dtype="string"), "a"]
)
@pytest.mark.parametrize("cmpop", _cmpops)
@pytest.mark.parametrize(
    "cmp_obj",
    [pd.Series(["b", "a", None, "d", "f", None], dtype="string"), "a"],
)
def test_string_series_compare(obj, cmpop, cmp_obj):

    g_obj = obj
    if isinstance(g_obj, pd.Series):
        g_obj = Series.from_pandas(g_obj)
    g_cmp_obj = cmp_obj
    if isinstance(g_cmp_obj, pd.Series):
        g_cmp_obj = Series.from_pandas(g_cmp_obj)
    got = cmpop(g_obj, g_cmp_obj)
    expected = cmpop(obj, cmp_obj)

    if isinstance(expected, pd.Series):
        expected = cudf.from_pandas(expected)

    utils.assert_eq(expected, got)


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("nelem", [1, 2, 100])
@pytest.mark.parametrize("cmpop", _cmpops)
@pytest.mark.parametrize("dtype", utils.NUMERIC_TYPES + ["datetime64[ms]"])
@pytest.mark.parametrize("use_cudf_scalar", [True, False])
def test_series_compare_scalar(
    nelem, cmpop, obj_class, dtype, use_cudf_scalar
):
    arr1 = np.random.randint(0, 100, 100).astype(dtype)
    sr1 = Series(arr1)
    rhs = random.choice(arr1).item()

    if use_cudf_scalar:
        rhs = cudf.Scalar(rhs)

    if obj_class == "Index":
        sr1 = as_index(sr1)

    result1 = cmpop(sr1, rhs)
    result2 = cmpop(rhs, sr1)

    if obj_class == "Index":
        result1 = Series(result1)
        result2 = Series(result2)

    np.testing.assert_equal(result1.to_array(), cmpop(arr1, rhs))
    np.testing.assert_equal(result2.to_array(), cmpop(rhs, arr1))


_nulls = ["none", "some"]


@pytest.mark.parametrize("nelem", [1, 7, 8, 9, 32, 64, 128])
@pytest.mark.parametrize("lhs_nulls,rhs_nulls", list(product(_nulls, _nulls)))
def test_validity_add(nelem, lhs_nulls, rhs_nulls):
    np.random.seed(0)
    # LHS
    lhs_data = np.random.random(nelem)
    if lhs_nulls == "some":
        lhs_mask = utils.random_bitmask(nelem)
        lhs_bitmask = utils.expand_bits_to_bytes(lhs_mask)[:nelem]
        lhs_null_count = utils.count_zero(lhs_bitmask)
        assert lhs_null_count >= 0
        lhs = Series.from_masked_array(lhs_data, lhs_mask)
        assert lhs.null_count == lhs_null_count
    else:
        lhs = Series(lhs_data)
    # RHS
    rhs_data = np.random.random(nelem)
    if rhs_nulls == "some":
        rhs_mask = utils.random_bitmask(nelem)
        rhs_bitmask = utils.expand_bits_to_bytes(rhs_mask)[:nelem]
        rhs_null_count = utils.count_zero(rhs_bitmask)
        assert rhs_null_count >= 0
        rhs = Series.from_masked_array(rhs_data, rhs_mask)
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
    got = res.fillna(na_value).to_array()
    expect = lhs_data + rhs_data
    if lhs_nulls == "some" or rhs_nulls == "some":
        expect[~res_mask] = na_value

    np.testing.assert_array_equal(expect, got)


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize(
    "binop,lhs_dtype,rhs_dtype",
    list(
        product(
            [operator.add, operator.mul],
            utils.NUMERIC_TYPES,
            utils.NUMERIC_TYPES,
        )
    ),
)
def test_series_binop_mixed_dtype(binop, lhs_dtype, rhs_dtype, obj_class):
    nelem = 10
    lhs = (np.random.random(nelem) * nelem).astype(lhs_dtype)
    rhs = (np.random.random(nelem) * nelem).astype(rhs_dtype)

    sr1 = Series(lhs)
    sr2 = Series(rhs)

    if obj_class == "Index":
        sr1 = as_index(sr1)
        sr2 = as_index(sr2)

    result = binop(Series(sr1), Series(sr2))

    if obj_class == "Index":
        result = Series(result)

    np.testing.assert_almost_equal(result.to_array(), binop(lhs, rhs))


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize(
    "cmpop,lhs_dtype,rhs_dtype",
    list(product(_cmpops, utils.NUMERIC_TYPES, utils.NUMERIC_TYPES)),
)
def test_series_cmpop_mixed_dtype(cmpop, lhs_dtype, rhs_dtype, obj_class):
    nelem = 5
    lhs = (np.random.random(nelem) * nelem).astype(lhs_dtype)
    rhs = (np.random.random(nelem) * nelem).astype(rhs_dtype)

    sr1 = Series(lhs)
    sr2 = Series(rhs)

    if obj_class == "Index":
        sr1 = as_index(sr1)
        sr2 = as_index(sr2)

    result = cmpop(Series(sr1), Series(sr2))

    if obj_class == "Index":
        result = Series(result)

    np.testing.assert_array_equal(result.to_array(), cmpop(lhs, rhs))


_reflected_ops = [
    lambda x: 1 + x,
    lambda x: 2 * x,
    lambda x: 2 - x,
    lambda x: 2 // x,
    lambda x: 2 / x,
    lambda x: 3 + x,
    lambda x: 3 * x,
    lambda x: 3 - x,
    lambda x: 3 // x,
    lambda x: 3 / x,
    lambda x: 3 % x,
    lambda x: -1 + x,
    lambda x: -2 * x,
    lambda x: -2 - x,
    lambda x: -2 // x,
    lambda x: -2 / x,
    lambda x: -3 + x,
    lambda x: -3 * x,
    lambda x: -3 - x,
    lambda x: -3 // x,
    lambda x: -3 / x,
    lambda x: -3 % x,
    lambda x: 0 + x,
    lambda x: 0 * x,
    lambda x: 0 - x,
    lambda x: 0 // x,
    lambda x: 0 / x,
]


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize(
    "func, dtype", list(product(_reflected_ops, utils.NUMERIC_TYPES))
)
def test_reflected_ops_scalar(func, dtype, obj_class):
    # create random series
    np.random.seed(12)
    random_series = utils.gen_rand(dtype, 100, low=10)

    # gpu series
    gs = Series(random_series)

    # class typing
    if obj_class == "Index":
        gs = as_index(gs)

    gs_result = func(gs)

    # class typing
    if obj_class == "Index":
        gs = Series(gs)

    # pandas
    ps_result = func(random_series)

    # verify
    np.testing.assert_allclose(ps_result, gs_result.to_array())


_cudf_scalar_reflected_ops = [
    lambda x: cudf.Scalar(1) + x,
    lambda x: cudf.Scalar(2) * x,
    lambda x: cudf.Scalar(2) - x,
    lambda x: cudf.Scalar(2) // x,
    lambda x: cudf.Scalar(2) / x,
    lambda x: cudf.Scalar(3) + x,
    lambda x: cudf.Scalar(3) * x,
    lambda x: cudf.Scalar(3) - x,
    lambda x: cudf.Scalar(3) // x,
    lambda x: cudf.Scalar(3) / x,
    lambda x: cudf.Scalar(3) % x,
    lambda x: cudf.Scalar(-1) + x,
    lambda x: cudf.Scalar(-2) * x,
    lambda x: cudf.Scalar(-2) - x,
    lambda x: cudf.Scalar(-2) // x,
    lambda x: cudf.Scalar(-2) / x,
    lambda x: cudf.Scalar(-3) + x,
    lambda x: cudf.Scalar(-3) * x,
    lambda x: cudf.Scalar(-3) - x,
    lambda x: cudf.Scalar(-3) // x,
    lambda x: cudf.Scalar(-3) / x,
    lambda x: cudf.Scalar(-3) % x,
    lambda x: cudf.Scalar(0) + x,
    lambda x: cudf.Scalar(0) * x,
    lambda x: cudf.Scalar(0) - x,
    lambda x: cudf.Scalar(0) // x,
    lambda x: cudf.Scalar(0) / x,
]


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize(
    "funcs, dtype",
    list(
        product(
            list(zip(_reflected_ops, _cudf_scalar_reflected_ops)),
            utils.NUMERIC_TYPES,
        )
    ),
)
def test_reflected_ops_cudf_scalar(funcs, dtype, obj_class):
    cpu_func, gpu_func = funcs

    # create random series
    np.random.seed(12)
    random_series = utils.gen_rand(dtype, 100, low=10)

    # gpu series
    gs = Series(random_series)

    # class typing
    if obj_class == "Index":
        gs = as_index(gs)

    gs_result = gpu_func(gs)

    # class typing
    if obj_class == "Index":
        gs = Series(gs)

    # pandas
    ps_result = cpu_func(random_series)

    # verify
    np.testing.assert_allclose(ps_result, gs_result.to_array())


@pytest.mark.parametrize("binop", _binops)
def test_different_shapes_and_columns(binop):

    # TODO: support `pow()` on NaN values. Particularly, the cases:
    #       `pow(1, NaN) == 1` and `pow(NaN, 0) == 1`
    if binop is operator.pow:
        return

    # Empty frame on the right side
    pd_frame = binop(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({}))
    cd_frame = binop(cudf.DataFrame({"x": [1, 2]}), cudf.DataFrame({}))
    utils.assert_eq(cd_frame, pd_frame)

    # Empty frame on the left side
    pd_frame = pd.DataFrame({}) + pd.DataFrame({"x": [1, 2]})
    cd_frame = cudf.DataFrame({}) + cudf.DataFrame({"x": [1, 2]})
    utils.assert_eq(cd_frame, pd_frame)

    # Note: the below rely on a discrepancy between cudf and pandas
    # While pandas inserts columns in alphabetical order, cudf inserts in the
    # order of whichever column comes first. So the following code will not
    # work if the names of columns are reversed i.e. ('y', 'x') != ('x', 'y')

    # More rows on the left side
    pd_frame = pd.DataFrame({"x": [1, 2, 3]}) + pd.DataFrame({"y": [1, 2]})
    cd_frame = cudf.DataFrame({"x": [1, 2, 3]}) + cudf.DataFrame({"y": [1, 2]})
    utils.assert_eq(cd_frame, pd_frame)

    # More rows on the right side
    pd_frame = pd.DataFrame({"x": [1, 2]}) + pd.DataFrame({"y": [1, 2, 3]})
    cd_frame = cudf.DataFrame({"x": [1, 2]}) + cudf.DataFrame({"y": [1, 2, 3]})
    utils.assert_eq(cd_frame, pd_frame)


@pytest.mark.parametrize("binop", _binops)
def test_different_shapes_and_same_columns(binop):

    # TODO: support `pow()` on NaN values. Particularly, the cases:
    #       `pow(1, NaN) == 1` and `pow(NaN, 0) == 1`
    if binop is operator.pow:
        return

    pd_frame = binop(
        pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [1, 2, 3]})
    )
    cd_frame = binop(
        cudf.DataFrame({"x": [1, 2]}), cudf.DataFrame({"x": [1, 2, 3]})
    )
    # cast x as float64 so it matches pandas dtype
    cd_frame["x"] = cd_frame["x"].astype(np.float64)
    utils.assert_eq(cd_frame, pd_frame)


@pytest.mark.parametrize("binop", _binops)
def test_different_shapes_and_columns_with_unaligned_indices(binop):

    # TODO: support `pow()` on NaN values. Particularly, the cases:
    #       `pow(1, NaN) == 1` and `pow(NaN, 0) == 1`
    if binop is operator.pow:
        return

    # Test with a RangeIndex
    pdf1 = pd.DataFrame({"x": [4, 3, 2, 1], "y": [7, 3, 8, 6]})
    # Test with a GenericIndex
    pdf2 = pd.DataFrame(
        {"x": [1, 2, 3, 7], "y": [4, 5, 6, 7]}, index=[0, 1, 3, 4]
    )
    # Test with a GenericIndex in a different order
    pdf3 = pd.DataFrame(
        {"x": [4, 5, 6, 7], "y": [1, 2, 3, 7], "z": [0, 5, 3, 7]},
        index=[0, 3, 5, 3],
    )
    gdf1 = cudf.DataFrame.from_pandas(pdf1)
    gdf2 = cudf.DataFrame.from_pandas(pdf2)
    gdf3 = cudf.DataFrame.from_pandas(pdf3)

    pd_frame = binop(binop(pdf1, pdf2), pdf3)
    cd_frame = binop(binop(gdf1, gdf2), gdf3)
    # cast x and y as float64 so it matches pandas dtype
    cd_frame["x"] = cd_frame["x"].astype(np.float64)
    cd_frame["y"] = cd_frame["y"].astype(np.float64)
    utils.assert_eq(cd_frame, pd_frame)


@pytest.mark.parametrize(
    "df2",
    [
        cudf.DataFrame({"a": [3, 2, 1]}, index=[3, 2, 1]),
        cudf.DataFrame([3, 2]),
    ],
)
@pytest.mark.parametrize("binop", [operator.eq, operator.ne])
def test_df_different_index_shape(df2, binop):
    df1 = cudf.DataFrame([1, 2, 3], index=[1, 2, 3])

    pdf1 = df1.to_pandas()
    pdf2 = df2.to_pandas()

    utils.assert_exceptions_equal(
        lfunc=binop,
        rfunc=binop,
        lfunc_args_and_kwargs=([pdf1, pdf2],),
        rfunc_args_and_kwargs=([df1, df2],),
    )


@pytest.mark.parametrize("op", [operator.eq, operator.ne])
def test_boolean_scalar_binop(op):
    psr = pd.Series(np.random.choice([True, False], 10))
    gsr = cudf.from_pandas(psr)
    utils.assert_eq(op(psr, True), op(gsr, True))
    utils.assert_eq(op(psr, False), op(gsr, False))

    # cuDF scalar
    utils.assert_eq(op(psr, True), op(gsr, cudf.Scalar(True)))
    utils.assert_eq(op(psr, False), op(gsr, cudf.Scalar(False)))


_operators_arithmetic = [
    "add",
    "radd",
    "sub",
    "rsub",
    "mul",
    "rmul",
    "mod",
    "rmod",
    "pow",
    "rpow",
    "floordiv",
    "rfloordiv",
    "truediv",
    "rtruediv",
]

_operators_comparison = ["eq", "ne", "lt", "le", "gt", "ge"]


@pytest.mark.parametrize("func", _operators_arithmetic)
@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("fill_value", [None, 27])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_operator_func_between_series(dtype, func, has_nulls, fill_value):
    count = 1000
    gdf_series_a = utils.gen_rand_series(
        dtype, count, has_nulls=has_nulls, stride=10000
    )
    gdf_series_b = utils.gen_rand_series(
        dtype, count, has_nulls=has_nulls, stride=100
    )
    pdf_series_a = gdf_series_a.to_pandas()
    pdf_series_b = gdf_series_b.to_pandas()

    gdf_result = getattr(gdf_series_a, func)(
        gdf_series_b, fill_value=fill_value
    )
    pdf_result = getattr(pdf_series_a, func)(
        pdf_series_b, fill_value=fill_value
    )

    utils.assert_eq(pdf_result, gdf_result)


@pytest.mark.parametrize("func", _operators_arithmetic)
@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("fill_value", [None, 27])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("use_cudf_scalar", [False, True])
def test_operator_func_series_and_scalar(
    dtype, func, has_nulls, fill_value, use_cudf_scalar
):
    count = 1000
    scalar = 59
    gdf_series = utils.gen_rand_series(
        dtype, count, has_nulls=has_nulls, stride=10000
    )
    pdf_series = gdf_series.to_pandas()

    gdf_series_result = getattr(gdf_series, func)(
        cudf.Scalar(scalar) if use_cudf_scalar else scalar,
        fill_value=fill_value,
    )
    pdf_series_result = getattr(pdf_series, func)(
        scalar, fill_value=fill_value
    )

    utils.assert_eq(pdf_series_result, gdf_series_result)


_permu_values = [0, 1, None, np.nan]


@pytest.mark.parametrize("fill_value", _permu_values)
@pytest.mark.parametrize("scalar_a", _permu_values)
@pytest.mark.parametrize("scalar_b", _permu_values)
@pytest.mark.parametrize("func", _operators_comparison)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_operator_func_between_series_logical(
    dtype, func, scalar_a, scalar_b, fill_value
):

    gdf_series_a = Series([scalar_a], nan_as_null=False).astype(dtype)
    gdf_series_b = Series([scalar_b], nan_as_null=False).astype(dtype)

    pdf_series_a = gdf_series_a.to_pandas(nullable=True)
    pdf_series_b = gdf_series_b.to_pandas(nullable=True)

    gdf_series_result = getattr(gdf_series_a, func)(
        gdf_series_b, fill_value=fill_value
    )
    pdf_series_result = getattr(pdf_series_a, func)(
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
            utils.assert_eq(expect, got)
        return
    utils.assert_eq(expect, got)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("func", _operators_comparison)
@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("scalar", [-59.0, np.nan, 0, 59.0])
@pytest.mark.parametrize("fill_value", [None, True, False, 1.0])
@pytest.mark.parametrize("use_cudf_scalar", [False, True])
def test_operator_func_series_and_scalar_logical(
    dtype, func, has_nulls, scalar, fill_value, use_cudf_scalar
):
    gdf_series = utils.gen_rand_series(
        dtype, 1000, has_nulls=has_nulls, stride=10000
    )
    pdf_series = gdf_series.to_pandas(nullable=True)
    gdf_series_result = getattr(gdf_series, func)(
        cudf.Scalar(scalar) if use_cudf_scalar else scalar,
        fill_value=fill_value,
    )
    pdf_series_result = getattr(pdf_series, func)(
        scalar, fill_value=fill_value
    )

    expect = pdf_series_result
    got = gdf_series_result.to_pandas(nullable=True)

    utils.assert_eq(expect, got)


@pytest.mark.parametrize("func", _operators_arithmetic)
@pytest.mark.parametrize("nulls", _nulls)
@pytest.mark.parametrize("fill_value", [None, 27])
@pytest.mark.parametrize("other", ["df", "scalar"])
def test_operator_func_dataframe(func, nulls, fill_value, other):
    num_rows = 100
    num_cols = 3

    def gen_df():
        pdf = pd.DataFrame()
        from string import ascii_lowercase

        cols = np.random.choice(num_cols + 5, num_cols, replace=False)

        for i in range(num_cols):
            colname = ascii_lowercase[cols[i]]
            data = utils.gen_rand("float64", num_rows) * 10000
            if nulls == "some":
                idx = np.random.choice(
                    num_rows, size=int(num_rows / 2), replace=False
                )
                data[idx] = np.nan
            pdf[colname] = data
        return pdf

    pdf1 = gen_df()
    pdf2 = gen_df() if other == "df" else 59.0
    gdf1 = cudf.DataFrame.from_pandas(pdf1)
    gdf2 = cudf.DataFrame.from_pandas(pdf2) if other == "df" else 59.0

    got = getattr(gdf1, func)(gdf2, fill_value=fill_value)
    expect = getattr(pdf1, func)(pdf2, fill_value=fill_value)[list(got._data)]

    utils.assert_eq(expect, got)


@pytest.mark.parametrize("func", _operators_arithmetic + _operators_comparison)
@pytest.mark.parametrize("rhs", [0, 1, 2, 128])
def test_binop_bool_uint(func, rhs):
    # TODO: remove this once issue #2172 is resolved
    if func == "rmod" or func == "rfloordiv":
        return
    psr = pd.Series([True, False, False])
    gsr = cudf.from_pandas(psr)
    utils.assert_eq(
        getattr(psr, func)(rhs), getattr(gsr, func)(rhs), check_dtype=False
    )


def test_series_misc_binop():
    pds = pd.Series([1, 2, 4], name="abc xyz")
    gds = cudf.Series([1, 2, 4], name="abc xyz")

    utils.assert_eq(pds + 1, gds + 1)
    utils.assert_eq(1 + pds, 1 + gds)

    utils.assert_eq(pds + pds, gds + gds)

    pds1 = pd.Series([1, 2, 4], name="hello world")
    gds1 = cudf.Series([1, 2, 4], name="hello world")

    utils.assert_eq(pds + pds1, gds + gds1)
    utils.assert_eq(pds1 + pds, gds1 + gds)

    utils.assert_eq(pds1 + pds + 5, gds1 + gds + 5)


def test_int8_float16_binop():
    a = cudf.Series([1], dtype="int8")
    b = np.float16(2)
    expect = cudf.Series([0.5])
    got = a / b
    utils.assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("dtype", ["int64", "float64", "str"])
def test_vector_to_none_binops(dtype):
    data = Series([1, 2, 3, None], dtype=dtype)

    expect = Series([None] * 4).astype(dtype)
    got = data + None

    utils.assert_eq(expect, got)


@pytest.mark.parametrize(
    "lhs",
    [
        1,
        3,
        4,
        pd.Series([5, 6, 2]),
        pd.Series([0, 10, 20, 30, 3, 4, 5, 6, 2]),
        6,
    ],
)
@pytest.mark.parametrize("rhs", [1, 3, 4, pd.Series([5, 6, 2])])
@pytest.mark.parametrize(
    "ops",
    [
        (np.remainder, cudf.remainder),
        (np.floor_divide, cudf.floor_divide),
        (np.subtract, cudf.subtract),
        (np.add, cudf.add),
        (np.true_divide, cudf.true_divide),
        (np.multiply, cudf.multiply),
    ],
)
def test_ufunc_ops(lhs, rhs, ops):
    np_op, cu_op = ops

    if isinstance(lhs, pd.Series):
        culhs = cudf.from_pandas(lhs)
    else:
        culhs = lhs

    if isinstance(rhs, pd.Series):
        curhs = cudf.from_pandas(rhs)
    else:
        curhs = rhs

    expect = np_op(lhs, rhs)
    got = cu_op(culhs, curhs)
    if np.isscalar(expect):
        assert got == expect
    else:
        utils.assert_eq(
            expect, got,
        )


def dtype_scalar(val, dtype):
    if dtype == "str":
        return str(val)
    dtype = np.dtype(dtype)
    if dtype.type in {np.datetime64, np.timedelta64}:
        res, _ = np.datetime_data(dtype)
        return dtype.type(val, res)
    else:
        return dtype.type(val)


def make_valid_scalar_add_data():
    valid = set()

    # to any int, we may add any kind of
    # other int, float, datetime timedelta, or bool
    valid |= set(
        product(
            INTEGER_TYPES,
            FLOAT_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES | BOOL_TYPES,
        )
    )

    # to any float, we may add any int, float, or bool
    valid |= set(
        product(FLOAT_TYPES, INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES)
    )

    # to any datetime, we may add any int, timedelta, or bool
    valid |= set(
        product(DATETIME_TYPES, INTEGER_TYPES | TIMEDELTA_TYPES | BOOL_TYPES)
    )

    # to any timedelta, we may add any int, datetime, other timedelta, or bool
    valid |= set(
        product(TIMEDELTA_TYPES, INTEGER_TYPES | DATETIME_TYPES | BOOL_TYPES)
    )

    # to any bool, we may add any int, float, datetime, timedelta, or bool
    valid |= set(
        product(
            BOOL_TYPES,
            INTEGER_TYPES
            | FLOAT_TYPES
            | DATETIME_TYPES
            | TIMEDELTA_TYPES
            | BOOL_TYPES,
        )
    )

    # to any string, we may add any other string
    valid |= {("str", "str")}

    return sorted(list(valid))


def make_invalid_scalar_add_data():
    invalid = set()

    # we can not add a datetime to a float
    invalid |= set(product(FLOAT_TYPES, DATETIME_TYPES))

    # We can not add a timedelta to a float
    invalid |= set(product(FLOAT_TYPES, TIMEDELTA_TYPES))

    # we can not add a float to any datetime
    invalid |= set(product(DATETIME_TYPES, FLOAT_TYPES))

    # can can not add a datetime to a datetime
    invalid |= set(product(DATETIME_TYPES, DATETIME_TYPES))

    # can not add a timedelta to a float
    invalid |= set(product(FLOAT_TYPES, TIMEDELTA_TYPES))

    return sorted(list(invalid))


@pytest.mark.parametrize("dtype_l,dtype_r", make_valid_scalar_add_data())
def test_scalar_add(dtype_l, dtype_r):
    test_value = 1

    lval_host = dtype_scalar(test_value, dtype=dtype_l)
    rval_host = dtype_scalar(test_value, dtype=dtype_r)

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    # expect = np.add(lval_host, rval_host)
    expect = lval_host + rval_host
    got = lval_gpu + rval_gpu

    assert expect == got.value
    if not dtype_l == dtype_r == "str":
        assert expect.dtype == got.dtype


@pytest.mark.parametrize("dtype_l,dtype_r", make_invalid_scalar_add_data())
def test_scalar_add_invalid(dtype_l, dtype_r):
    test_value = 1

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    with pytest.raises(TypeError):
        lval_gpu + rval_gpu


def make_scalar_difference_data():
    valid = set()

    # from an int, we may subtract any int, float, timedelta,
    # or boolean value
    valid |= set(
        product(
            INTEGER_TYPES,
            INTEGER_TYPES | FLOAT_TYPES | TIMEDELTA_TYPES | BOOL_TYPES,
        )
    )

    # from any float, we may subtract any int, float, or bool
    valid |= set(
        product(FLOAT_TYPES, INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES)
    )

    # from any datetime we may subtract any int, datetime, timedelta, or bool
    valid |= set(
        product(
            DATETIME_TYPES,
            INTEGER_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES | BOOL_TYPES,
        )
    )

    # from any timedelta we may subtract any int, timedelta, or bool
    valid |= set(
        product(TIMEDELTA_TYPES, INTEGER_TYPES | TIMEDELTA_TYPES | BOOL_TYPES)
    )

    # from any bool we may subtract any int, float or timedelta
    valid |= set(
        product(BOOL_TYPES, INTEGER_TYPES | FLOAT_TYPES | TIMEDELTA_TYPES)
    )

    return sorted(list(valid))


def make_scalar_difference_data_invalid():
    invalid = set()

    # we can't subtract a datetime from an int
    invalid |= set(product(INTEGER_TYPES, DATETIME_TYPES))

    # we can't subtract a datetime or timedelta from a float
    invalid |= set(product(FLOAT_TYPES, DATETIME_TYPES | TIMEDELTA_TYPES))

    # we can't subtract a float from a datetime or timedelta
    invalid |= set(product(DATETIME_TYPES | TIMEDELTA_TYPES, FLOAT_TYPES))

    # We can't subtract a datetime from a timedelta
    invalid |= set(product(TIMEDELTA_TYPES, DATETIME_TYPES))

    # we can't subtract a datetime or bool from a bool
    invalid |= set(product(BOOL_TYPES, BOOL_TYPES | DATETIME_TYPES))

    return sorted(list(invalid))


@pytest.mark.parametrize("dtype_l,dtype_r", make_scalar_difference_data())
def test_scalar_difference(dtype_l, dtype_r):
    test_value = 1

    lval_host = dtype_scalar(test_value, dtype=dtype_l)
    rval_host = dtype_scalar(test_value, dtype=dtype_r)

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    expect = lval_host - rval_host
    got = lval_gpu - rval_gpu

    assert expect == got.value
    assert expect.dtype == got.dtype


@pytest.mark.parametrize(
    "dtype_l,dtype_r", make_scalar_difference_data_invalid()
)
def test_scalar_difference_invalid(dtype_l, dtype_r):
    test_value = 1

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    with pytest.raises(TypeError):
        lval_gpu - rval_gpu


def make_scalar_product_data():
    valid = set()

    # we can multiply an int, or bool by any int, float, timedelta, or bool
    valid |= set(
        product(
            INTEGER_TYPES | BOOL_TYPES,
            INTEGER_TYPES | FLOAT_TYPES | TIMEDELTA_TYPES | BOOL_TYPES,
        )
    )

    # we can muliply any timedelta by any int, or bool
    valid |= set(product(TIMEDELTA_TYPES, INTEGER_TYPES | BOOL_TYPES))

    # we can multiply a float by any int, float, or bool
    valid |= set(
        product(FLOAT_TYPES, INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES)
    )

    return sorted(list(valid))


def make_scalar_product_data_invalid():
    invalid = set()

    # can't multiply a ints, floats, datetimes, timedeltas,
    # or bools by datetimes
    invalid |= set(
        product(
            INTEGER_TYPES
            | FLOAT_TYPES
            | DATETIME_TYPES
            | TIMEDELTA_TYPES
            | BOOL_TYPES,
            DATETIME_TYPES,
        )
    )

    # can't multiply datetimes with anything really
    invalid |= set(
        product(
            DATETIME_TYPES,
            INTEGER_TYPES
            | FLOAT_TYPES
            | DATETIME_TYPES
            | TIMEDELTA_TYPES
            | BOOL_TYPES,
        )
    )

    # can't multiply timedeltas by timedeltas
    invalid |= set(product(TIMEDELTA_TYPES, TIMEDELTA_TYPES))

    return sorted(list(invalid))


@pytest.mark.parametrize("dtype_l,dtype_r", make_scalar_product_data())
def test_scalar_product(dtype_l, dtype_r):
    test_value = 1

    lval_host = dtype_scalar(test_value, dtype=dtype_l)
    rval_host = dtype_scalar(test_value, dtype=dtype_r)

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    expect = lval_host * rval_host
    got = lval_gpu * rval_gpu

    assert expect == got.value
    assert expect.dtype == got.dtype


@pytest.mark.parametrize("dtype_l,dtype_r", make_scalar_product_data_invalid())
def test_scalar_product_invalid(dtype_l, dtype_r):
    test_value = 1

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    with pytest.raises(TypeError):
        lval_gpu * rval_gpu


def make_scalar_floordiv_data():
    valid = set()

    # we can divide ints and floats by other ints, floats, or bools
    valid |= set(
        product(
            INTEGER_TYPES | FLOAT_TYPES,
            INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES,
        )
    )

    # we can divide timedeltas by ints, floats or other timedeltas
    valid |= set(
        product(TIMEDELTA_TYPES, INTEGER_TYPES | FLOAT_TYPES | TIMEDELTA_TYPES)
    )

    # we can divide bools by ints, floats or bools
    valid |= set(product(BOOL_TYPES, INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES))

    return sorted(list(valid))


def make_scalar_floordiv_data_invalid():
    invalid = set()

    # we can't numeric types into datelike types
    invalid |= set(
        product(
            INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES,
            DATETIME_TYPES | TIMEDELTA_TYPES,
        )
    )

    # we can't divide datetime types into anything
    invalid |= set(
        product(
            DATETIME_TYPES,
            INTEGER_TYPES
            | FLOAT_TYPES
            | DATETIME_TYPES
            | TIMEDELTA_TYPES
            | BOOL_TYPES,
        )
    )

    # we can't divide timedeltas into bools, or datetimes
    invalid |= set(product(TIMEDELTA_TYPES, BOOL_TYPES | DATETIME_TYPES))

    return sorted(list(invalid))


@pytest.mark.parametrize("dtype_l,dtype_r", make_scalar_floordiv_data())
def test_scalar_floordiv(dtype_l, dtype_r):
    test_value = 1

    lval_host = dtype_scalar(test_value, dtype=dtype_l)
    rval_host = dtype_scalar(test_value, dtype=dtype_r)

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    expect = lval_host // rval_host
    got = lval_gpu // rval_gpu

    assert expect == got.value
    assert expect.dtype == got.dtype


@pytest.mark.parametrize(
    "dtype_l,dtype_r", make_scalar_floordiv_data_invalid()
)
def test_scalar_floordiv_invalid(dtype_l, dtype_r):
    test_value = 1

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    with pytest.raises(TypeError):
        lval_gpu // rval_gpu


def make_scalar_truediv_data():
    valid = set()

    # we can true divide ints, floats, or bools by other
    # ints, floats or bools
    valid |= set(
        product(
            INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES,
            INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES,
        )
    )

    # we can true divide timedeltas by ints floats or timedeltas
    valid |= set(product(TIMEDELTA_TYPES, INTEGER_TYPES | TIMEDELTA_TYPES))

    return sorted(list(valid))


def make_scalar_truediv_data_invalid():
    invalid = set()

    # we can't divide ints, floats or bools by datetimes
    # or timedeltas
    invalid |= set(
        product(
            INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES,
            DATETIME_TYPES | TIMEDELTA_TYPES,
        )
    )

    # we cant true divide datetime types by anything
    invalid |= set(
        product(
            DATETIME_TYPES,
            INTEGER_TYPES
            | FLOAT_TYPES
            | DATETIME_TYPES
            | TIMEDELTA_TYPES
            | BOOL_TYPES,
        )
    )

    # we cant true divide timedeltas by datetimes or bools or floats
    invalid |= set(
        product(TIMEDELTA_TYPES, DATETIME_TYPES | BOOL_TYPES | FLOAT_TYPES)
    )

    return sorted(list(invalid))


@pytest.mark.parametrize("dtype_l,dtype_r", make_scalar_truediv_data())
def test_scalar_truediv(dtype_l, dtype_r):
    test_value = 1

    lval_host = dtype_scalar(test_value, dtype=dtype_l)
    rval_host = dtype_scalar(test_value, dtype=dtype_r)

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    expect = np.true_divide(lval_host, rval_host)
    got = lval_gpu / rval_gpu

    assert expect == got.value

    # numpy bug

    if np.dtype(dtype_l).itemsize <= 2 and np.dtype(dtype_r).itemsize <= 2:
        assert expect.dtype == "float64" and got.dtype == "float32"
    else:
        assert expect.dtype == got.dtype
    # assert expect.dtype == got.dtype


@pytest.mark.parametrize("dtype_l,dtype_r", make_scalar_truediv_data_invalid())
def test_scalar_truediv_invalid(dtype_l, dtype_r):
    test_value = 1

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    with pytest.raises(TypeError):
        lval_gpu / rval_gpu


def make_scalar_remainder_data():
    valid = set()

    # can mod numeric types with each other
    valid |= set(
        product(
            INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES,
            INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES,
        )
    )

    # can mod timedeltas by other timedeltas
    valid |= set(product(TIMEDELTA_TYPES, TIMEDELTA_TYPES))

    return sorted(list(valid))


def make_scalar_remainder_data_invalid():
    invalid = set()

    # numeric types cant be modded against timedeltas
    # or datetimes. Also, datetimes can't be modded
    # against datetimes or timedeltas
    invalid |= set(
        product(
            INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES | DATETIME_TYPES,
            DATETIME_TYPES | TIMEDELTA_TYPES,
        )
    )

    # datetime and timedelta types cant be modded against
    # any numeric types
    invalid |= set(
        product(
            DATETIME_TYPES | TIMEDELTA_TYPES,
            INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES,
        )
    )

    # timedeltas cant mod with datetimes
    invalid |= set(product(TIMEDELTA_TYPES, DATETIME_TYPES))

    return sorted(list(invalid))


@pytest.mark.parametrize("dtype_l,dtype_r", make_scalar_remainder_data())
def test_scalar_remainder(dtype_l, dtype_r):
    test_value = 1

    lval_host = dtype_scalar(test_value, dtype=dtype_l)
    rval_host = dtype_scalar(test_value, dtype=dtype_r)

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    expect = lval_host % rval_host
    got = lval_gpu % rval_gpu

    assert expect == got.value
    assert expect.dtype == got.dtype


@pytest.mark.parametrize(
    "dtype_l,dtype_r", make_scalar_remainder_data_invalid()
)
def test_scalar_remainder_invalid(dtype_l, dtype_r):
    test_value = 1

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    with pytest.raises(TypeError):
        lval_gpu % rval_gpu


def make_scalar_power_data():
    # only numeric values form valid operands for power
    return sorted(
        product(
            INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES,
            INTEGER_TYPES | FLOAT_TYPES | BOOL_TYPES,
        )
    )


def make_scalar_power_data_invalid():
    invalid = set()

    # datetimes and timedeltas cant go in exponents
    invalid |= set(
        product(
            INTEGER_TYPES
            | FLOAT_TYPES
            | TIMEDELTA_TYPES
            | DATETIME_TYPES
            | BOOL_TYPES,
            DATETIME_TYPES | TIMEDELTA_TYPES,
        )
    )

    # datetimes and timedeltas may not be raised to
    # any exponent of any dtype
    invalid |= set(
        product(
            DATETIME_TYPES | TIMEDELTA_TYPES,
            DATETIME_TYPES
            | TIMEDELTA_TYPES
            | INTEGER_TYPES
            | FLOAT_TYPES
            | BOOL_TYPES,
        )
    )

    return sorted(list(invalid))


@pytest.mark.parametrize("dtype_l,dtype_r", make_scalar_power_data())
def test_scalar_power(dtype_l, dtype_r):
    test_value = 1

    lval_host = dtype_scalar(test_value, dtype=dtype_l)
    rval_host = dtype_scalar(test_value, dtype=dtype_r)

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    expect = lval_host ** rval_host
    got = lval_gpu ** rval_gpu

    assert expect == got.value
    assert expect.dtype == got.dtype


@pytest.mark.parametrize("dtype_l,dtype_r", make_scalar_power_data_invalid())
def test_scalar_power_invalid(dtype_l, dtype_r):
    test_value = 1

    lval_gpu = cudf.Scalar(test_value, dtype=dtype_l)
    rval_gpu = cudf.Scalar(test_value, dtype=dtype_r)

    with pytest.raises(TypeError):
        lval_gpu ** rval_gpu


@pytest.mark.parametrize(
    "date_col",
    [
        [
            "2000-01-01 00:00:00.012345678",
            "2000-01-31 00:00:00.012345678",
            "2000-02-29 00:00:00.012345678",
        ]
    ],
)
@pytest.mark.parametrize("n_periods", [0, 1, -1, 12, -12])
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
        pytest.param(
            "nanoseconds",
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/36589"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    ["datetime64[ns]", "datetime64[us]", "datetime64[ms]", "datetime64[s]"],
)
@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_datetime_dateoffset_binaryop(
    date_col, n_periods, frequency, dtype, op
):
    gsr = cudf.Series(date_col, dtype=dtype)
    psr = gsr.to_pandas()  # converts to nanos

    kwargs = {frequency: n_periods}

    goffset = cudf.DateOffset(**kwargs)
    poffset = pd.DateOffset(**kwargs)

    expect = op(psr, poffset)
    got = op(gsr, goffset)

    utils.assert_eq(expect, got)

    expect = op(psr, -poffset)
    got = op(gsr, -goffset)

    utils.assert_eq(expect, got)


@pytest.mark.parametrize(
    "date_col",
    [
        [
            "2000-01-01 00:00:00.012345678",
            "2000-01-31 00:00:00.012345678",
            "2000-02-29 00:00:00.012345678",
        ]
    ],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"months": 2, "years": 5},
        {"microseconds": 1, "seconds": 1},
        {"months": 2, "years": 5, "seconds": 923, "microseconds": 481},
        pytest.param(
            {"milliseconds": 4},
            marks=pytest.mark.xfail(
                reason="Pandas gets the wrong answer for milliseconds"
            ),
        ),
        pytest.param(
            {"milliseconds": 4, "years": 2},
            marks=pytest.mark.xfail(
                reason="Pandas construction fails with these keywords"
            ),
        ),
        pytest.param(
            {"nanoseconds": 12},
            marks=pytest.mark.xfail(
                reason="Pandas gets the wrong answer for nanoseconds"
            ),
        ),
    ],
)
@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_datetime_dateoffset_binaryop_multiple(date_col, kwargs, op):

    gsr = cudf.Series(date_col, dtype="datetime64[ns]")
    psr = gsr.to_pandas()

    poffset = pd.DateOffset(**kwargs)
    goffset = cudf.DateOffset(**kwargs)

    expect = op(psr, poffset)
    got = op(gsr, goffset)

    utils.assert_eq(expect, got)


@pytest.mark.parametrize(
    "date_col",
    [
        [
            "2000-01-01 00:00:00.012345678",
            "2000-01-31 00:00:00.012345678",
            "2000-02-29 00:00:00.012345678",
        ]
    ],
)
@pytest.mark.parametrize("n_periods", [0, 1, -1, 12, -12])
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
        pytest.param(
            "nanoseconds",
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/36589"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    ["datetime64[ns]", "datetime64[us]", "datetime64[ms]", "datetime64[s]"],
)
def test_datetime_dateoffset_binaryop_reflected(
    date_col, n_periods, frequency, dtype
):
    gsr = cudf.Series(date_col, dtype=dtype)
    psr = gsr.to_pandas()  # converts to nanos

    kwargs = {frequency: n_periods}

    goffset = cudf.DateOffset(**kwargs)
    poffset = pd.DateOffset(**kwargs)

    expect = poffset + psr
    got = goffset + gsr

    utils.assert_eq(expect, got)

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
        val = np.dtype(dtype).type(4, "s")
    elif dtype == "timedelta64[s]":
        val = np.dtype(dtype).type(4, "s")
    elif dtype == "category":
        val = np.int64(4)
    else:
        val = np.dtype(dtype).type(4)

    expected = val == data.to_pandas()
    got = val == data

    # In case of index, expected would be a numpy array
    if isinstance(data, cudf.BaseIndex):
        expected = pd.Index(expected)

    utils.assert_eq(expected, got)


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "datetime64[ns]",
        "datetime64[us]",
        "datetime64[ms]",
        "datetime64[s]",
        "timedelta64[ns]",
        "timedelta64[us]",
        "timedelta64[ms]",
        "timedelta64[s]",
    ],
)
@pytest.mark.parametrize("op", _operators_comparison)
def test_binops_with_NA_consistent(dtype, op):
    data = [1, 2, 3]
    sr = cudf.Series(data, dtype=dtype)

    result = getattr(sr, op)(cudf.NA)
    if dtype in NUMERIC_TYPES:
        if op == "ne":
            expect_all = True
        else:
            expect_all = False
        assert (result == expect_all).all()
    elif dtype in DATETIME_TYPES & TIMEDELTA_TYPES:
        assert result._column.null_count == len(data)


@pytest.mark.parametrize(
    "args",
    [
        (
            operator.add,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["3.0", "4.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
        ),
        (
            operator.add,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
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
            cudf.Decimal64Dtype(scale=3, precision=18),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", "0.995"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
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
            cudf.Decimal64Dtype(scale=6, precision=18),
        ),
        (
            operator.mul,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["1.5", "3.0"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["2.25", "6.0"],
            cudf.Decimal64Dtype(scale=5, precision=7),
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
            operator.truediv,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=4),
            ["1.5", "3.0"],
            cudf.Decimal64Dtype(scale=1, precision=4),
            ["1.0", "0.6"],
            cudf.Decimal64Dtype(scale=1, precision=9),
        ),
        (
            operator.truediv,
            ["110", "200"],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=2, precision=4),
            ["1000.0", "1000.0"],
            cudf.Decimal64Dtype(scale=-3, precision=8),
        ),
        (
            operator.truediv,
            ["132.86", "15.25"],
            cudf.Decimal64Dtype(scale=4, precision=14),
            ["2.34", "8.50"],
            cudf.Decimal64Dtype(scale=2, precision=8),
            ["56.77", "1.79"],
            cudf.Decimal64Dtype(scale=2, precision=18),
        ),
        (
            operator.add,
            ["1.5", None, "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["1.5", None, "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["3.0", None, "4.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
        ),
        (
            operator.add,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["3.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.mul,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["1.5", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=5, precision=7),
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=10),
            ["0.1", None],
            cudf.Decimal64Dtype(scale=3, precision=12),
            ["10.0", None],
            cudf.Decimal64Dtype(scale=1, precision=18),
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
def test_binops_decimal(args):
    op, lhs, l_dtype, rhs, r_dtype, expect, expect_dtype = args

    a = utils._decimal_series(lhs, l_dtype)
    b = utils._decimal_series(rhs, r_dtype)
    expect = (
        utils._decimal_series(expect, expect_dtype)
        if isinstance(expect_dtype, cudf.Decimal64Dtype)
        else cudf.Series(expect, dtype=expect_dtype)
    )

    got = op(a, b)
    assert expect.dtype == got.dtype
    utils.assert_eq(expect, got)


@pytest.mark.parametrize(
    "args",
    [
        (
            operator.eq,
            ["100", "41", None],
            cudf.Decimal64Dtype(scale=0, precision=5),
            [100, 42, 12],
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.eq,
            ["100.000", "42.001", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 12],
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.eq,
            ["100", "40", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 12],
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.ne,
            ["100", "42", "24", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 40, 24, 12],
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, True, False, None], dtype=bool),
        ),
        (
            operator.ne,
            ["10.1", "88", "11", None],
            cudf.Decimal64Dtype(scale=1, precision=3),
            [10, 42, 11, 12],
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, True, False, None], dtype=bool),
        ),
        (
            operator.ne,
            ["100.000", "42", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            cudf.Series([False, False, True, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100", "40", "28", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 42, 24, 12],
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            cudf.Series([False, False, True, None], dtype=bool),
            cudf.Series([False, True, False, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100", "42", "20", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 40, 24, 12],
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            cudf.Series([False, False, True, None], dtype=bool),
            cudf.Series([False, True, False, None], dtype=bool),
        ),
        (
            operator.le,
            ["100", "40", "28", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 42, 24, 12],
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.le,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            cudf.Series([True, False, True, None], dtype=bool),
            cudf.Series([True, True, False, None], dtype=bool),
        ),
        (
            operator.le,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100", "42", "20", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 40, 24, 12],
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            cudf.Series([True, False, True, None], dtype=bool),
            cudf.Series([True, True, False, None], dtype=bool),
        ),
    ],
)
@pytest.mark.parametrize("integer_dtype", utils.INTEGER_TYPES)
@pytest.mark.parametrize("reflected", [True, False])
def test_binops_decimal_comp_mixed_integer(args, integer_dtype, reflected):
    """
    Tested compare operations:
        eq, lt, gt, le, ge
    Each operation has 3 decimal data setups, with scale from {==0, >0, <0}.
    Decimal precisions are sufficient to hold the digits.
    For each decimal data setup, there is at least one row that lead to one
    of the following compare results: {True, False, None}.
    """
    if not reflected:
        op, ldata, ldtype, rdata, expected, _ = args
    else:
        op, ldata, ldtype, rdata, _, expected = args

    lhs = utils._decimal_series(ldata, ldtype)
    rhs = cudf.Series(rdata, dtype=integer_dtype)

    if reflected:
        rhs, lhs = lhs, rhs

    actual = op(lhs, rhs)

    utils.assert_eq(expected, actual)


@pytest.mark.parametrize(
    "args",
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
            cudf.Scalar(decimal.Decimal("1.5")),
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
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            cudf.Scalar(decimal.Decimal("1.5")),
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
            cudf.Scalar(decimal.Decimal("1.5")),
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
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            cudf.Scalar(decimal.Decimal("1.5")),
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
            cudf.Decimal64Dtype(scale=-2, precision=6),
            False,
        ),
        (
            operator.truediv,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=2, precision=4),
            decimal.Decimal(2),
            ["50", "100"],
            cudf.Decimal64Dtype(scale=2, precision=6),
            False,
        ),
        (
            operator.truediv,
            ["35.23", "54.91"],
            cudf.Decimal64Dtype(scale=2, precision=4),
            decimal.Decimal("1.5"),
            ["23.4", "36.6"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            False,
        ),
        (
            operator.truediv,
            ["22.2", "93.6"],
            cudf.Decimal64Dtype(scale=1, precision=3),
            cudf.Scalar(decimal.Decimal("1.5")),
            ["14", "62"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.truediv,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            1,
            ["0", "0"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            True,
        ),
        (
            operator.truediv,
            ["1.2", "0.5"],
            cudf.Decimal64Dtype(scale=1, precision=6),
            decimal.Decimal(20),
            ["10", "40"],
            cudf.Decimal64Dtype(scale=-1, precision=9),
            True,
        ),
        (
            operator.truediv,
            ["1.22", "5.24"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            decimal.Decimal("8.55"),
            ["7", "1"],
            cudf.Decimal64Dtype(scale=0, precision=7),
            True,
        ),
        (
            operator.truediv,
            ["1.1", "42.8"],
            cudf.Decimal64Dtype(scale=1, precision=3),
            cudf.Scalar(decimal.Decimal("90.84")),
            ["82.5", "2.1"],
            cudf.Decimal64Dtype(scale=1, precision=8),
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
            cudf.Scalar(decimal.Decimal("2.5")),
            ["97.5", "197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
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
            cudf.Scalar(decimal.Decimal("2.5")),
            ["-97.5", "-197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            True,
        ),
    ],
)
def test_binops_decimal_scalar(args):
    op, lhs, l_dtype, rhs, expect, expect_dtype, reflect = args

    def decimal_series(input, dtype):
        return cudf.Series(
            [x if x is None else decimal.Decimal(x) for x in input],
            dtype=dtype,
        )

    lhs = decimal_series(lhs, l_dtype)
    expect = decimal_series(expect, expect_dtype)

    if reflect:
        lhs, rhs = rhs, lhs

    got = op(lhs, rhs)
    assert expect.dtype == got.dtype
    utils.assert_eq(expect, got)


@pytest.mark.parametrize(
    "args",
    [
        (
            operator.eq,
            ["100.00", "41", None],
            cudf.Decimal64Dtype(scale=0, precision=5),
            100,
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.eq,
            ["100.123", "41", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.eq,
            ["100.123", "41", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            cudf.Scalar(decimal.Decimal("100.123")),
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.ne,
            ["100.00", "41", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            cudf.Series([False, True, None], dtype=bool),
            cudf.Series([False, True, None], dtype=bool),
        ),
        (
            operator.ne,
            ["100.123", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            cudf.Series([False, True, None], dtype=bool),
            cudf.Series([False, True, None], dtype=bool),
        ),
        (
            operator.ne,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            cudf.Scalar(decimal.Decimal("100.123")),
            cudf.Series([False, True, True, None], dtype=bool),
            cudf.Series([False, True, True, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            cudf.Series([False, False, True, None], dtype=bool),
            cudf.Series([False, True, False, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            cudf.Series([False, False, True, None], dtype=bool),
            cudf.Series([False, True, False, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            cudf.Scalar(decimal.Decimal("100.123")),
            cudf.Series([False, False, True, None], dtype=bool),
            cudf.Series([False, True, False, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            cudf.Series([True, False, True, None], dtype=bool),
            cudf.Series([True, True, False, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            cudf.Series([True, False, True, None], dtype=bool),
            cudf.Series([True, True, False, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            cudf.Scalar(decimal.Decimal("100.123")),
            cudf.Series([True, False, True, None], dtype=bool),
            cudf.Series([True, True, False, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            cudf.Scalar(decimal.Decimal("100.123")),
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.le,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.le,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.le,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            cudf.Scalar(decimal.Decimal("100.123")),
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
    ],
)
@pytest.mark.parametrize("reflected", [True, False])
def test_binops_decimal_scalar_compare(args, reflected):
    """
    Tested compare operations:
        eq, lt, gt, le, ge
    Each operation has 3 data setups: pyints, Decimal, and
    decimal cudf.Scalar
    For each data setup, there is at least one row that lead to one of the
    following compare results: {True, False, None}.
    """
    if not reflected:
        op, ldata, ldtype, rdata, expected, _ = args
    else:
        op, ldata, ldtype, rdata, _, expected = args

    lhs = utils._decimal_series(ldata, ldtype)
    rhs = rdata

    if reflected:
        rhs, lhs = lhs, rhs

    actual = op(lhs, rhs)

    utils.assert_eq(expected, actual)


@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "str",
        "datetime64[ns]",
        "datetime64[us]",
        "datetime64[ms]",
        "datetime64[s]",
        "timedelta64[ns]",
        "timedelta64[us]",
        "timedelta64[ms]",
        "timedelta64[s]",
    ],
)
@pytest.mark.parametrize("null_scalar", [None, cudf.NA, np.datetime64("NaT")])
@pytest.mark.parametrize("cmpop", _cmpops)
def test_column_null_scalar_comparison(dtype, null_scalar, cmpop):
    # This test is meant to validate that comparing
    # a series of any dtype with a null scalar produces
    # a new series where all the elements are <NA>.

    if isinstance(null_scalar, np.datetime64):
        if np.dtype(dtype).kind not in "mM":
            pytest.skip()
        null_scalar = null_scalar.astype(dtype)

    dtype = np.dtype(dtype)

    data = [1, 2, 3, 4, 5]
    sr = cudf.Series(data, dtype=dtype)
    result = cmpop(sr, null_scalar)

    assert result.isnull().all()


@pytest.mark.parametrize("fn", ["eq", "ne", "lt", "gt", "le", "ge"])
def test_equality_ops_index_mismatch(fn):
    a = cudf.Series(
        [1, 2, 3, None, None, 4], index=["a", "b", "c", "d", "e", "f"]
    )
    b = cudf.Series(
        [-5, 4, 3, 2, 1, 0, 19, 11],
        index=["aa", "b", "c", "d", "e", "f", "y", "z"],
    )

    pa = a.to_pandas(nullable=True)
    pb = b.to_pandas(nullable=True)
    expected = getattr(pa, fn)(pb)
    actual = getattr(a, fn)(b).to_pandas(nullable=True)

    utils.assert_eq(expected, actual)


def generate_test_null_equals_columnops_data():
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

    null_cases = ["neither", "left", "right", "both"]
    data = [1, 2, 3]

    results = []
    # TODO: Numeric types can be cross compared as null equal
    for dtype in (
        list(NUMERIC_TYPES)
        + list(DATETIME_TYPES)
        + list(TIMEDELTA_TYPES)
        + list(STRING_TYPES)
        + ["category"]
    ):
        for case in null_cases:
            left = cudf.Series(data, dtype=dtype)
            right = cudf.Series(data, dtype=dtype)
            if case in {"left", "right"}:
                answer = False
            else:
                answer = True
            left, right = set_null_cases(left, right, case)
            results.append((left._column, right._column, answer, case))

    return results


@pytest.mark.parametrize(
    "lcol,rcol,ans,case", generate_test_null_equals_columnops_data()
)
def test_null_equals_columnops(lcol, rcol, ans, case):
    assert lcol._null_equals(rcol).all() == ans

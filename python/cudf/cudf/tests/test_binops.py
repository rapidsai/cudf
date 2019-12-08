# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division

import operator
import random
from itertools import product

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core import Series
from cudf.core.index import as_index
from cudf.tests import utils

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


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("nelem,binop", list(product([1, 2, 100], _binops)))
def test_series_binop_scalar(nelem, binop, obj_class):
    arr = np.random.random(nelem)
    rhs = random.choice(arr).item()
    sr = Series(arr)
    if obj_class == "Index":
        sr = as_index(sr)

    result = binop(sr, rhs)

    if obj_class == "Index":
        result = Series(result)

    np.testing.assert_almost_equal(result.to_array(), binop(arr, rhs))


_bitwise_binops = [operator.and_, operator.or_, operator.xor]


_int_types = ["int8", "int16", "int32", "int64", "longlong"]


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


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("nelem", [1, 2, 100])
@pytest.mark.parametrize("cmpop", _cmpops)
@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "datetime64[ms]",
        "longlong",
    ],
)
def test_series_compare_scalar(nelem, cmpop, obj_class, dtype):
    arr1 = np.random.randint(0, 100, 100).astype(dtype)
    sr1 = Series(arr1)
    rhs = random.choice(arr1).item()

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
            utils.expand_bits_to_bytes(lhs_mask & rhs_mask), dtype=np.bool
        )[:nelem]
    if lhs_nulls == "some" and rhs_nulls == "none":
        res_mask = np.asarray(
            utils.expand_bits_to_bytes(lhs_mask), dtype=np.bool
        )[:nelem]
    if lhs_nulls == "none" and rhs_nulls == "some":
        res_mask = np.asarray(
            utils.expand_bits_to_bytes(rhs_mask), dtype=np.bool
        )[:nelem]
    # Fill NA values
    na_value = -10000
    got = res.fillna(na_value).to_array()
    expect = lhs_data + rhs_data
    if lhs_nulls == "some" or rhs_nulls == "some":
        expect[~res_mask] = na_value

    np.testing.assert_array_equal(expect, got)


_dtypes = [np.int16, np.int32, np.int64, np.float32, np.float64]


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize(
    "binop,lhs_dtype,rhs_dtype",
    list(product([operator.add, operator.mul], _dtypes, _dtypes)),
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
    "cmpop,lhs_dtype,rhs_dtype", list(product(_cmpops, _dtypes, _dtypes))
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
@pytest.mark.parametrize("func, dtype", list(product(_reflected_ops, _dtypes)))
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
    np.testing.assert_allclose(ps_result, gs_result)


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


@pytest.mark.parametrize("op", [operator.eq, operator.ne])
def test_boolean_scalar_binop(op):
    psr = pd.Series(np.random.choice([True, False], 10))
    gsr = cudf.from_pandas(psr)
    utils.assert_eq(op(psr, True), op(gsr, True))
    utils.assert_eq(op(psr, False), op(gsr, False))


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
def test_operator_func_series_and_scalar(dtype, func, has_nulls, fill_value):
    count = 1000
    scalar = 59
    gdf_series = utils.gen_rand_series(
        dtype, count, has_nulls=has_nulls, stride=10000
    )
    pdf_series = gdf_series.to_pandas()

    gdf_series_result = getattr(gdf_series, func)(
        scalar, fill_value=fill_value
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
    gdf_series_a = Series([scalar_a]).astype(dtype)
    gdf_series_b = Series([scalar_b]).astype(dtype)
    pdf_series_a = gdf_series_a.to_pandas()
    pdf_series_b = gdf_series_b.to_pandas()

    gdf_series_result = getattr(gdf_series_a, func)(
        gdf_series_b, fill_value=fill_value
    )
    pdf_series_result = getattr(pdf_series_a, func)(
        pdf_series_b, fill_value=fill_value
    )

    if scalar_a in [None, np.nan] and scalar_b in [None, np.nan]:
        # cudf binary operations will return `None` when both left- and right-
        # side values are `None`. It will return `np.nan` when either side is
        # `np.nan`. As a consequence, when we convert our gdf => pdf during
        # assert_eq, we get a pdf with dtype='object' (all inputs are none).
        # to account for this, we use fillna.
        gdf_series_result.fillna(func == "ne", inplace=True)

    utils.assert_eq(pdf_series_result, gdf_series_result)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("func", _operators_comparison)
@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("scalar", [-59.0, np.nan, 0, 59.0])
@pytest.mark.parametrize("fill_value", [None, True, False, 1.0])
def test_operator_func_series_and_scalar_logical(
    dtype, func, has_nulls, scalar, fill_value
):
    gdf_series = utils.gen_rand_series(
        dtype, 1000, has_nulls=has_nulls, stride=10000
    )
    pdf_series = gdf_series.to_pandas()

    gdf_series_result = getattr(gdf_series, func)(
        scalar, fill_value=fill_value
    )
    pdf_series_result = getattr(pdf_series, func)(
        scalar, fill_value=fill_value
    )

    utils.assert_eq(pdf_series_result, gdf_series_result)


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
    expect = getattr(pdf1, func)(pdf2, fill_value=fill_value)[list(got._cols)]

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

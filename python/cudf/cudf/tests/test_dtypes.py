# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import cudf
from cudf.core.column import ColumnBase
from cudf.core.dtypes import (
    CategoricalDtype,
    Decimal32Dtype,
    Decimal64Dtype,
    Decimal128Dtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)
from cudf.testing._utils import assert_eq
from cudf.utils.dtypes import np_to_pa_dtype


def test_cdt_basic():
    psr = pd.Series(["a", "b", "a", "c"], dtype="category")
    sr = cudf.Series(["a", "b", "a", "c"], dtype="category")
    assert isinstance(sr.dtype, CategoricalDtype)
    assert_eq(sr.dtype.categories, psr.dtype.categories)


@pytest.mark.parametrize(
    "data", [None, [], ["a"], [1], [1.0], ["a", "b", "c"]]
)
@pytest.mark.parametrize("ordered", [None, False, True])
def test_cdt_eq(data, ordered):
    dt = cudf.CategoricalDtype(categories=data, ordered=ordered)
    assert dt == "category"
    assert dt == dt
    assert dt == cudf.CategoricalDtype(categories=None, ordered=ordered)
    assert dt == cudf.CategoricalDtype(categories=data, ordered=ordered)
    assert not dt == cudf.CategoricalDtype(
        categories=data, ordered=not ordered
    )


@pytest.mark.parametrize(
    "data", [None, [], ["a"], [1], [1.0], ["a", "b", "c"]]
)
@pytest.mark.parametrize("ordered", [None, False, True])
def test_cdf_to_pandas(data, ordered):
    assert (
        pd.CategoricalDtype(data, ordered)
        == cudf.CategoricalDtype(categories=data, ordered=ordered).to_pandas()
    )


@pytest.mark.parametrize(
    "value_type",
    [
        int,
        "int32",
        np.int32,
        "datetime64[ms]",
        "datetime64[ns]",
        "str",
        "object",
    ],
)
def test_list_dtype_pyarrow_round_trip(value_type):
    pa_type = pa.list_(cudf.utils.dtypes.np_to_pa_dtype(np.dtype(value_type)))
    expect = pa_type
    got = ListDtype.from_arrow(expect).to_arrow()
    assert expect.equals(got)


def test_list_dtype_eq():
    lhs = ListDtype("int32")
    rhs = ListDtype("int32")
    assert lhs == rhs
    rhs = ListDtype("int64")
    assert lhs != rhs


def test_list_nested_dtype():
    dt = ListDtype(ListDtype("int32"))
    expect = ListDtype("int32")
    got = dt.element_type
    assert expect == got


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"a": "int64"},
        {"a": "datetime64[ms]"},
        {"a": "int32", "b": "int64"},
    ],
)
def test_struct_dtype_pyarrow_round_trip(fields):
    pa_type = pa.struct(
        {
            k: cudf.utils.dtypes.np_to_pa_dtype(np.dtype(v))
            for k, v in fields.items()
        }
    )
    expect = pa_type
    got = StructDtype.from_arrow(expect).to_arrow()
    assert expect.equals(got)


def test_struct_dtype_eq():
    lhs = StructDtype(
        {"a": "int32", "b": StructDtype({"c": "int64", "ab": "int32"})}
    )
    rhs = StructDtype(
        {"a": "int32", "b": StructDtype({"c": "int64", "ab": "int32"})}
    )
    assert lhs == rhs
    rhs = StructDtype({"a": "int32", "b": "int64"})
    assert lhs != rhs
    lhs = StructDtype({"b": "int64", "a": "int32"})
    assert lhs != rhs


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"a": "int32"},
        {"a": "object"},
        {"a": "str"},
        {"a": "datetime64[D]"},
        {"a": "int32", "b": "int64"},
        {"a": "int32", "b": StructDtype({"a": "int32", "b": "int64"})},
    ],
)
def test_struct_dtype_fields(fields):
    fields = {"a": "int32", "b": StructDtype({"c": "int64", "d": "int32"})}
    dt = StructDtype(fields)
    assert_eq(dt.fields, fields)


@pytest.mark.parametrize(
    "decimal_type",
    [cudf.Decimal32Dtype, cudf.Decimal64Dtype, cudf.Decimal128Dtype],
)
def test_decimal_dtype_arrow_roundtrip(decimal_type):
    dt = decimal_type(4, 2)
    assert dt.to_arrow() == pa.decimal128(4, 2)
    assert dt == decimal_type.from_arrow(pa.decimal128(4, 2))


@pytest.mark.parametrize(
    "decimal_type,max_precision",
    [
        (cudf.Decimal32Dtype, 9),
        (cudf.Decimal64Dtype, 18),
        (cudf.Decimal128Dtype, 38),
    ],
)
def test_max_precision(decimal_type, max_precision):
    decimal_type(scale=0, precision=max_precision)
    with pytest.raises(ValueError):
        decimal_type(scale=0, precision=max_precision + 1)


@pytest.fixture(params=["int64", "int32"])
def subtype(request):
    return request.param


@pytest.fixture(params=["left", "right", "both", "neither"])
def closed(request):
    return request.param


def test_interval_dtype_pyarrow_round_trip(subtype, closed):
    pa_array = ArrowIntervalType(subtype, closed)
    expect = pa_array
    got = IntervalDtype.from_arrow(expect).to_arrow()
    assert expect.equals(got)


def test_interval_dtype_from_pandas(subtype, closed):
    expect = cudf.IntervalDtype(subtype, closed=closed)
    pd_type = pd.IntervalDtype(subtype, closed=closed)
    got = cudf.IntervalDtype.from_pandas(pd_type)
    assert expect == got


def assert_column_array_dtype_equal(column: ColumnBase, array: pa.array):
    """
    In cudf, each column holds its dtype. And since column may have child
    columns, child columns also holds their datatype. This method tests
    that every level of `column` matches the type of the given `array`
    recursively.
    """

    if isinstance(column.dtype, ListDtype):
        return array.type.equals(
            column.dtype.to_arrow()
        ) and assert_column_array_dtype_equal(
            column.base_children[1], array.values
        )
    elif isinstance(column.dtype, StructDtype):
        return array.type.equals(column.dtype.to_arrow()) and all(
            assert_column_array_dtype_equal(child, array.field(i))
            for i, child in enumerate(column.base_children)
        )
    elif isinstance(
        column.dtype, (Decimal128Dtype, Decimal64Dtype, Decimal32Dtype)
    ):
        return array.type.equals(column.dtype.to_arrow())
    elif isinstance(column.dtype, CategoricalDtype):
        raise NotImplementedError()
    else:
        return array.type.equals(np_to_pa_dtype(column.dtype))


@pytest.mark.parametrize(
    "data",
    [
        [[{"name": 123}]],
        [
            [
                {
                    "IsLeapYear": False,
                    "data": {"Year": 1999, "Month": 7},
                    "names": ["Mike", None],
                },
                {
                    "IsLeapYear": True,
                    "data": {"Year": 2004, "Month": 12},
                    "names": None,
                },
                {
                    "IsLeapYear": False,
                    "data": {"Year": 1996, "Month": 2},
                    "names": ["Rose", "Richard"],
                },
            ]
        ],
        [
            [None, {"human?": True, "deets": {"weight": 2.4, "age": 27}}],
            [
                {"human?": None, "deets": {"weight": 5.3, "age": 25}},
                {"human?": False, "deets": {"weight": 8.0, "age": 31}},
                {"human?": False, "deets": None},
            ],
            [],
            None,
            [{"human?": None, "deets": {"weight": 6.9, "age": None}}],
        ],
        [
            {
                "name": "var0",
                "val": [
                    {"name": "var1", "val": None, "type": "optional<struct>"}
                ],
                "type": "list",
            },
            {},
            {
                "name": "var2",
                "val": [
                    {
                        "name": "var3",
                        "val": {"field": 42},
                        "type": "optional<struct>",
                    },
                    {
                        "name": "var4",
                        "val": {"field": 3.14},
                        "type": "optional<struct>",
                    },
                ],
                "type": "list",
            },
            None,
        ],
    ],
)
def test_lists_of_structs_dtype(data):
    got = cudf.Series(data)
    expected = pa.array(data)

    assert_column_array_dtype_equal(got._column, expected)
    assert expected.equals(got._column.to_arrow())


@pytest.mark.parametrize(
    "in_dtype,expect",
    [
        (np.dtype("int8"), np.dtype("int8")),
        (np.int8, np.dtype("int8")),
        (pd.Int8Dtype(), np.dtype("int8")),
        (pd.StringDtype(), np.dtype("object")),
        ("int8", np.dtype("int8")),
        ("boolean", np.dtype("bool")),
        ("bool_", np.dtype("bool")),
        (np.bool_, np.dtype("bool")),
        (int, np.dtype("int64")),
        (float, np.dtype("float64")),
        (cudf.ListDtype("int64"), cudf.ListDtype("int64")),
        (np.dtype("U"), np.dtype("object")),
        ("timedelta64[ns]", np.dtype("<m8[ns]")),
        ("timedelta64[ms]", np.dtype("<m8[ms]")),
        ("<m8[s]", np.dtype("<m8[s]")),
        ("datetime64[ns]", np.dtype("<M8[ns]")),
        ("datetime64[ms]", np.dtype("<M8[ms]")),
        ("<M8[s]", np.dtype("<M8[s]")),
        (cudf.ListDtype("int64"), cudf.ListDtype("int64")),
        ("category", cudf.CategoricalDtype()),
        (
            cudf.CategoricalDtype(categories=("a", "b", "c")),
            cudf.CategoricalDtype(categories=("a", "b", "c")),
        ),
        (
            pd.CategoricalDtype(categories=("a", "b", "c")),
            cudf.CategoricalDtype(categories=("a", "b", "c")),
        ),
        (
            # this is a pandas.core.arrays.numpy_.PandasDtype...
            pd.array([1], dtype="int16").dtype,
            np.dtype("int16"),
        ),
        (pd.IntervalDtype("int"), cudf.IntervalDtype("int64")),
        (cudf.IntervalDtype("int"), cudf.IntervalDtype("int64")),
        (pd.IntervalDtype("int64"), cudf.IntervalDtype("int64")),
    ],
)
def test_dtype(in_dtype, expect):
    assert_eq(cudf.dtype(in_dtype), expect)


@pytest.mark.parametrize(
    "in_dtype",
    [
        "complex",
        np.complex128,
        complex,
        "S",
        "a",
        "V",
        "float16",
        np.float16,
        "timedelta64",
        "timedelta64[D]",
        "datetime64[D]",
        "datetime64",
    ],
)
def test_dtype_raise(in_dtype):
    with pytest.raises(TypeError):
        cudf.dtype(in_dtype)


def test_dtype_np_bool_to_pa_bool():
    """This test case captures that utility np_to_pa_dtype
    should map np.bool_ to pa.bool_, nuances on bit width
    difference should be handled elsewhere.
    """

    assert np_to_pa_dtype(np.dtype("bool")) == pa.bool_()

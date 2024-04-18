# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc


@pytest.fixture(
    scope="module",
    params=[
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.float32(),
        pa.float64(),
    ],
    ids=[
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
    ],
)
def numeric_pa_type(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        pa.timestamp("s"),
        pa.timestamp("ms"),
        pa.timestamp("us"),
        pa.timestamp("ns"),
    ],
    ids=["s", "ms", "us", "ns"],
)
def timestamp_pa_type(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        pa.duration("s"),
        pa.duration("ms"),
        pa.duration("us"),
        pa.duration("ns"),
    ],
    ids=["s", "ms", "us", "ns"],
)
def duration_pa_type(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        plc.column_factories.MaskState.UNALLOCATED,
        plc.column_factories.MaskState.ALL_VALID,
        plc.column_factories.MaskState.ALL_NULL,
        plc.column_factories.MaskState.UNINITIALIZED,
    ],
    ids=["unallocated", "all_valid", "all_null", "uninitialized"],
)
def mask_state(request):
    return request.param


def test_make_empty_column_dtype(pa_type):
    pa_col = pa.array([], type=pa_type)

    # TODO: DataType.from_arrow()?
    plc_type = plc.interop.from_arrow(pa_col).type()

    if isinstance(pa_type, (pa.ListType, pa.StructType)):
        with pytest.raises(ValueError):
            plc.column_factories.make_empty_column(plc_type)
        return

    cudf_col = plc.column_factories.make_empty_column(plc_type)
    assert_column_eq(cudf_col, pa_col)


def test_make_empty_column_typeid(pa_type):
    pa_col = pa.array([], type=pa_type)

    # TODO: DataType.from_arrow()?
    tid = plc.interop.from_arrow(pa_col).type().id()

    if isinstance(pa_type, (pa.ListType, pa.StructType)):
        with pytest.raises(ValueError):
            plc.column_factories.make_empty_column(tid)
        return

    cudf_col = plc.column_factories.make_empty_column(tid)
    assert_column_eq(cudf_col, pa_col)


def test_make_numeric_column(numeric_pa_type, mask_state):
    size = 3

    if mask_state == plc.column_factories.MaskState.ALL_NULL:
        expected = pa.array([None] * size, type=numeric_pa_type)
    else:
        # TODO: uninitialized not necessarily 0
        expected = pa.array([0] * size, type=numeric_pa_type)

    plc_type = plc.interop.from_arrow(
        pa.array([], type=numeric_pa_type)
    ).type()

    got = plc.column_factories.make_numeric_column(plc_type, size, mask_state)
    assert_column_eq(got, expected)


def test_make_fixed_point_column(mask_state):
    size = 3
    scale = 2
    precision = 38  # libcudf drops precision

    if mask_state == plc.column_factories.MaskState.ALL_NULL:
        expected = pa.array(
            [None] * size, type=pa.decimal128(precision, scale)
        )
    else:
        expected = pa.array([0] * size, type=pa.decimal128(precision, scale))

    plc_type = plc.interop.from_arrow(
        pa.array([], type=pa.decimal128(precision, scale))
    ).type()
    got = plc.column_factories.make_fixed_point_column(
        plc_type, size, mask_state
    )
    assert_column_eq(got, expected)


def test_make_timestamp_column(timestamp_pa_type, mask_state):
    size = 3

    if mask_state == plc.column_factories.MaskState.ALL_NULL:
        expected = pa.array([None] * size, type=timestamp_pa_type)
    else:
        expected = pa.array([0] * size, type=timestamp_pa_type)

    plc_type = plc.interop.from_arrow(
        pa.array([], type=timestamp_pa_type)
    ).type()

    got = plc.column_factories.make_timestamp_column(
        plc_type, size, mask_state
    )
    assert_column_eq(got, expected)


def test_make_duration_column(duration_pa_type, mask_state):
    size = 3

    if mask_state == plc.column_factories.MaskState.ALL_NULL:
        expected = pa.array([None] * size, type=duration_pa_type)
    else:
        expected = pa.array([0] * size, type=duration_pa_type)

    plc_type = plc.interop.from_arrow(
        pa.array([], type=duration_pa_type)
    ).type()

    got = plc.column_factories.make_duration_column(plc_type, size, mask_state)
    assert_column_eq(got, expected)

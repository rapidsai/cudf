# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import DEFAULT_STRUCT_TESTING_TYPE, assert_column_eq

from cudf._lib import pylibcudf as plc

size = 3

NUMERIC_TYPES = [
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
    pa.bool_(),
]

TIMESTAMP_TYPES = [
    pa.timestamp("s"),
    pa.timestamp("ms"),
    pa.timestamp("us"),
    pa.timestamp("ns"),
]

DURATION_TYPES = [
    pa.duration("s"),
    pa.duration("ms"),
    pa.duration("us"),
    pa.duration("ns"),
]

DECIMAL_TYPES = [pa.decimal128(38, 2)]

STRING_TYPES = [pa.string()]
STRUCT_TYPES = [DEFAULT_STRUCT_TESTING_TYPE]
LIST_TYPES = [pa.list_(pa.int64())]

ALL_TYPES = (
    NUMERIC_TYPES
    + TIMESTAMP_TYPES
    + DURATION_TYPES
    + STRING_TYPES
    + DECIMAL_TYPES
    + STRUCT_TYPES
    + LIST_TYPES
)


def pa_type_to_plc_type(pa_type):
    # TODO: should be a cleaner way
    return plc.interop.from_arrow(pa.array([], type=pa_type)).type()


@pytest.fixture(scope="module", params=NUMERIC_TYPES, ids=repr)
def numeric_pa_type(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=DECIMAL_TYPES,
    ids=repr,
)
def fixed_point_pa_type(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=TIMESTAMP_TYPES,
    ids=repr,
)
def timestamp_pa_type(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=DURATION_TYPES,
    ids=repr,
)
def duration_pa_type(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        plc.MaskState.UNALLOCATED,
        plc.MaskState.ALL_VALID,
        plc.MaskState.ALL_NULL,
        plc.MaskState.UNINITIALIZED,
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
    if mask_state == plc.column_factories.MaskState.ALL_NULL:
        expected = pa.array([None] * size, type=numeric_pa_type)
    else:
        # TODO: uninitialized not necessarily 0
        expected = pa.array(
            [0 if numeric_pa_type is not pa.bool_() else False] * size,
            type=numeric_pa_type,
        )

    plc_type = pa_type_to_plc_type(numeric_pa_type)

    got = plc.column_factories.make_numeric_column(plc_type, size, mask_state)
    assert_column_eq(got, expected)


@pytest.mark.parametrize(
    "non_numeric_pa_type", list(set(ALL_TYPES) - set(NUMERIC_TYPES))
)
def test_make_numeric_column_dtype_err(non_numeric_pa_type):
    plc_type = pa_type_to_plc_type(non_numeric_pa_type)
    with pytest.raises(ValueError):
        plc.column_factories.make_numeric_column(
            plc_type, 3, plc.column_factories.MaskState.UNALLOCATED
        )


def test_make_numeric_column_negative_size_err(numeric_pa_type):
    plc_type = pa_type_to_plc_type(numeric_pa_type)
    with pytest.raises(RuntimeError):
        plc.column_factories.make_numeric_column(
            plc_type, -1, plc.column_factories.MaskState.UNALLOCATED
        )


def test_make_fixed_point_column(fixed_point_pa_type, mask_state):
    if mask_state == plc.column_factories.MaskState.ALL_NULL:
        expected = pa.array([None] * size, type=fixed_point_pa_type)
    else:
        expected = pa.array([0] * size, type=fixed_point_pa_type)

    plc_type = pa_type_to_plc_type(fixed_point_pa_type)

    got = plc.column_factories.make_fixed_point_column(
        plc_type, size, mask_state
    )
    assert_column_eq(got, expected)


@pytest.mark.parametrize(
    "non_fixed_point_pa_type", list(set(ALL_TYPES) - set(DECIMAL_TYPES))
)
def test_make_fixed_point_column_dtype_err(non_fixed_point_pa_type):
    plc_type = pa_type_to_plc_type(non_fixed_point_pa_type)
    with pytest.raises(ValueError):
        plc.column_factories.make_fixed_point_column(
            plc_type, 3, plc.column_factories.MaskState.UNALLOCATED
        )


def test_make_fixed_point_column_negative_size_err(fixed_point_pa_type):
    plc_type = pa_type_to_plc_type(fixed_point_pa_type)
    with pytest.raises(RuntimeError):
        plc.column_factories.make_fixed_point_column(
            plc_type, -1, plc.column_factories.MaskState.UNALLOCATED
        )


def test_make_timestamp_column(timestamp_pa_type, mask_state):
    if mask_state == plc.column_factories.MaskState.ALL_NULL:
        expected = pa.array([None] * size, type=timestamp_pa_type)
    else:
        expected = pa.array([0] * size, type=timestamp_pa_type)

    plc_type = pa_type_to_plc_type(timestamp_pa_type)

    got = plc.column_factories.make_timestamp_column(
        plc_type, size, mask_state
    )
    assert_column_eq(got, expected)


@pytest.mark.parametrize(
    "non_timestamp_pa_type", list(set(ALL_TYPES) - set(TIMESTAMP_TYPES))
)
def test_make_timestamp_column_dtype_err(non_timestamp_pa_type):
    plc_type = pa_type_to_plc_type(non_timestamp_pa_type)
    with pytest.raises(ValueError):
        plc.column_factories.make_timestamp_column(
            plc_type, 3, plc.column_factories.MaskState.UNALLOCATED
        )


def test_make_timestamp_column_negative_size_err(timestamp_pa_type):
    plc_type = pa_type_to_plc_type(timestamp_pa_type)
    with pytest.raises(RuntimeError):
        plc.column_factories.make_timestamp_column(
            plc_type, -1, plc.column_factories.MaskState.UNALLOCATED
        )


def test_make_duration_column(duration_pa_type, mask_state):
    if mask_state == plc.column_factories.MaskState.ALL_NULL:
        expected = pa.array([None] * size, type=duration_pa_type)
    else:
        expected = pa.array([0] * size, type=duration_pa_type)

    plc_type = pa_type_to_plc_type(duration_pa_type)

    got = plc.column_factories.make_duration_column(plc_type, size, mask_state)
    assert_column_eq(got, expected)


@pytest.mark.parametrize(
    "non_duration_pa_type", list(set(ALL_TYPES) - set(DURATION_TYPES))
)
def test_make_duration_column_dtype_err(non_duration_pa_type):
    plc_type = pa_type_to_plc_type(non_duration_pa_type)
    with pytest.raises(ValueError):
        plc.column_factories.make_duration_column(
            plc_type, 3, plc.column_factories.MaskState.UNALLOCATED
        )


def test_make_duration_column_negative_size_err(duration_pa_type):
    plc_type = pa_type_to_plc_type(duration_pa_type)
    with pytest.raises(RuntimeError):
        plc.column_factories.make_duration_column(
            plc_type, -1, plc.column_factories.MaskState.UNALLOCATED
        )

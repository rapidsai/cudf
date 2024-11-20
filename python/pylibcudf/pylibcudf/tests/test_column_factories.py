# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import DEFAULT_STRUCT_TESTING_TYPE, assert_column_eq

import pylibcudf as plc

EMPTY_COL_SIZE = 3

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

    plc_type = plc.interop.from_arrow(pa_col).type()

    if isinstance(pa_type, (pa.ListType, pa.StructType)):
        with pytest.raises(TypeError):
            plc.column_factories.make_empty_column(plc_type)
        return

    cudf_col = plc.column_factories.make_empty_column(plc_type)
    assert_column_eq(cudf_col, pa_col)


def test_make_empty_column_typeid(pa_type):
    pa_col = pa.array([], type=pa_type)

    tid = plc.interop.from_arrow(pa_col).type().id()

    if isinstance(pa_type, (pa.ListType, pa.StructType)):
        with pytest.raises(TypeError):
            plc.column_factories.make_empty_column(tid)
        return

    cudf_col = plc.column_factories.make_empty_column(tid)
    assert_column_eq(cudf_col, pa_col)


def validate_empty_column(col, mask_state, dtype):
    assert col.size() == EMPTY_COL_SIZE

    if mask_state == plc.types.MaskState.UNALLOCATED:
        assert col.null_count() == 0
    elif mask_state == plc.types.MaskState.ALL_VALID:
        assert col.null_count() == 0
    elif mask_state == plc.types.MaskState.ALL_NULL:
        assert col.null_count() == EMPTY_COL_SIZE

    assert plc.interop.to_arrow(col).type == dtype


def test_make_numeric_column(numeric_pa_type, mask_state):
    plc_type = plc.interop.from_arrow(numeric_pa_type)

    got = plc.column_factories.make_numeric_column(
        plc_type, EMPTY_COL_SIZE, mask_state
    )
    validate_empty_column(got, mask_state, numeric_pa_type)


@pytest.mark.parametrize(
    "non_numeric_pa_type", [t for t in ALL_TYPES if t not in NUMERIC_TYPES]
)
def test_make_numeric_column_dtype_err(non_numeric_pa_type):
    plc_type = plc.interop.from_arrow(non_numeric_pa_type)
    with pytest.raises(TypeError):
        plc.column_factories.make_numeric_column(
            plc_type, 3, plc.types.MaskState.UNALLOCATED
        )


def test_make_numeric_column_negative_size_err(numeric_pa_type):
    plc_type = plc.interop.from_arrow(numeric_pa_type)
    with pytest.raises(RuntimeError):
        plc.column_factories.make_numeric_column(
            plc_type, -1, plc.types.MaskState.UNALLOCATED
        )


def test_make_fixed_point_column(fixed_point_pa_type, mask_state):
    plc_type = plc.interop.from_arrow(fixed_point_pa_type)

    got = plc.column_factories.make_fixed_point_column(
        plc_type, EMPTY_COL_SIZE, mask_state
    )

    validate_empty_column(got, mask_state, fixed_point_pa_type)


@pytest.mark.parametrize(
    "non_fixed_point_pa_type", [t for t in ALL_TYPES if t not in DECIMAL_TYPES]
)
def test_make_fixed_point_column_dtype_err(non_fixed_point_pa_type):
    plc_type = plc.interop.from_arrow(non_fixed_point_pa_type)
    with pytest.raises(TypeError):
        plc.column_factories.make_fixed_point_column(
            plc_type, 3, plc.types.MaskState.UNALLOCATED
        )


def test_make_fixed_point_column_negative_size_err(fixed_point_pa_type):
    plc_type = plc.interop.from_arrow(fixed_point_pa_type)
    with pytest.raises(RuntimeError):
        plc.column_factories.make_fixed_point_column(
            plc_type, -1, plc.types.MaskState.UNALLOCATED
        )


def test_make_timestamp_column(timestamp_pa_type, mask_state):
    plc_type = plc.interop.from_arrow(timestamp_pa_type)

    got = plc.column_factories.make_timestamp_column(
        plc_type, EMPTY_COL_SIZE, mask_state
    )
    validate_empty_column(got, mask_state, timestamp_pa_type)


@pytest.mark.parametrize(
    "non_timestamp_pa_type", [t for t in ALL_TYPES if t not in TIMESTAMP_TYPES]
)
def test_make_timestamp_column_dtype_err(non_timestamp_pa_type):
    plc_type = plc.interop.from_arrow(non_timestamp_pa_type)
    with pytest.raises(TypeError):
        plc.column_factories.make_timestamp_column(
            plc_type, 3, plc.types.MaskState.UNALLOCATED
        )


def test_make_timestamp_column_negative_size_err(timestamp_pa_type):
    plc_type = plc.interop.from_arrow(timestamp_pa_type)
    with pytest.raises(RuntimeError):
        plc.column_factories.make_timestamp_column(
            plc_type, -1, plc.types.MaskState.UNALLOCATED
        )


def test_make_duration_column(duration_pa_type, mask_state):
    plc_type = plc.interop.from_arrow(duration_pa_type)

    got = plc.column_factories.make_duration_column(
        plc_type, EMPTY_COL_SIZE, mask_state
    )
    validate_empty_column(got, mask_state, duration_pa_type)


@pytest.mark.parametrize(
    "non_duration_pa_type", [t for t in ALL_TYPES if t not in DURATION_TYPES]
)
def test_make_duration_column_dtype_err(non_duration_pa_type):
    plc_type = plc.interop.from_arrow(non_duration_pa_type)
    with pytest.raises(TypeError):
        plc.column_factories.make_duration_column(
            plc_type, 3, plc.types.MaskState.UNALLOCATED
        )


def test_make_duration_column_negative_size_err(duration_pa_type):
    plc_type = plc.interop.from_arrow(duration_pa_type)
    with pytest.raises(RuntimeError):
        plc.column_factories.make_duration_column(
            plc_type, -1, plc.types.MaskState.UNALLOCATED
        )

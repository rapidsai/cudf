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
)
def numeric_pa_type(request):
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


def test_make_numeric_column_no_mask(numeric_pa_type):
    size = 3
    expected = pa.array([0] * size, type=numeric_pa_type)
    plc_type = plc.interop.from_arrow(
        pa.array([], type=numeric_pa_type)
    ).type()

    got = plc.column_factories.make_numeric_column(
        plc_type, size, plc.column_factories.MaskState.UNALLOCATED
    )
    assert_column_eq(got, expected)


def test_make_numeric_column_all_valid(numeric_pa_type):
    size = 3
    expected = pa.array([0] * size, type=numeric_pa_type)
    plc_type = plc.interop.from_arrow(
        pa.array([], type=numeric_pa_type)
    ).type()

    got = plc.column_factories.make_numeric_column(
        plc_type, size, plc.column_factories.MaskState.ALL_VALID
    )
    assert_column_eq(got, expected)


def test_make_numeric_column_all_null(numeric_pa_type):
    size = 3
    expected = pa.array([None] * size, type=numeric_pa_type)
    plc_type = plc.interop.from_arrow(
        pa.array([], type=numeric_pa_type)
    ).type()

    got = plc.column_factories.make_numeric_column(
        plc_type, size, plc.column_factories.MaskState.ALL_NULL
    )
    assert_column_eq(got, expected)


def test_make_numeric_column_uninit_mask(numeric_pa_type):
    size = 3
    expected = pa.array([0] * size, type=numeric_pa_type)
    plc_type = plc.interop.from_arrow(
        pa.array([], type=numeric_pa_type)
    ).type()

    got = plc.column_factories.make_numeric_column(
        plc_type, size, plc.column_factories.MaskState.UNINITIALIZED
    )
    assert_column_eq(got, expected)

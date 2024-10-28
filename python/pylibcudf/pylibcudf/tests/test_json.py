# Copyright (c) 2024, NVIDIA CORPORATION.
import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def plc_col():
    arr = pa.array(
        ['{"foo": {"bar": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]', None]
    )
    return plc.interop.from_arrow(arr)


@pytest.fixture(scope="module")
def json_path():
    slr = pa.scalar("$.foo.bar")
    return plc.interop.from_arrow(slr)


@pytest.mark.parametrize("allow_single_quotes", [True, False])
@pytest.mark.parametrize("strip_quotes_from_single_strings", [True, False])
@pytest.mark.parametrize("missing_fields_as_nulls", [True, False])
def test_get_json_object(
    plc_col,
    json_path,
    allow_single_quotes,
    strip_quotes_from_single_strings,
    missing_fields_as_nulls,
):
    result = plc.json.get_json_object(
        plc_col,
        json_path,
        plc.json.GetJsonObjectOptions(
            allow_single_quotes=allow_single_quotes,
            strip_quotes_from_single_strings=strip_quotes_from_single_strings,
            missing_fields_as_nulls=missing_fields_as_nulls,
        ),
    )
    expected = pa.array(['[{"a": 1, "b": 2}, {"a": 3, "b": 4}]', None])
    assert_column_eq(result, expected)

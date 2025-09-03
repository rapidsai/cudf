# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.mark.parametrize(
    "source_func",
    [
        "make_source",
        "make_source_from_file",
    ],
)
@pytest.mark.parametrize("options", [None, plc.io.text.ParseOptions()])
def test_multibyte_split(source_func, options, tmp_path):
    data = "x::y::z"
    func = getattr(plc.io.text, source_func)
    if source_func == "make_source":
        source = func(data)
    elif source_func == "make_source_from_file":
        fle = tmp_path / "fle.txt"
        fle.write_text(data)
        source = func(str(fle))
    result = plc.io.text.multibyte_split(source, "::", options)
    expected = pa.array(["x::", "y::", "z"])
    assert_column_eq(result, expected)

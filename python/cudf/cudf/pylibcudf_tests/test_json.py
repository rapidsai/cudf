# Copyright (c) 2024, NVIDIA CORPORATION.
import io
import os
import pathlib

import pandas as pd
import pytest

import cudf._lib.pylibcudf as plc


@pytest.fixture(
    params=["a.txt", pathlib.Path("a.txt"), io.BytesIO(), io.StringIO()],
)
def sink(request):
    yield request.param
    # Cleanup after ourselves
    # since the BytesIO and StringIO objects get cached by pytest
    if isinstance(request.param, io.IOBase):
        buf = request.param
        buf.seek(0)
        buf.truncate(0)


@pytest.mark.parametrize("lines", [True, False])
def test_write_json_basic(table_data, sink, tmp_path, lines):
    plc_table_w_meta, pa_table = table_data
    if isinstance(sink, str):
        sink = f"{tmp_path}/{sink}"
    elif isinstance(sink, os.PathLike):
        sink = tmp_path.joinpath(sink)
    plc.io.json.write_json(
        plc.io.SinkInfo([sink]), plc_table_w_meta, lines=lines
    )

    # orient=records (basically what the cudf json writer does,
    # doesn't preserve colnames when there are zero rows in table)
    exp = pa_table.to_pandas()

    if len(exp) == 0:
        exp = pd.DataFrame()

    # Convert everything to string to make
    # comparisons easier

    if isinstance(sink, (str, os.PathLike)):
        with open(sink, "r") as f:
            str_result = f.read()
    elif isinstance(sink, io.BytesIO):
        sink.seek(0)
        str_result = sink.read().decode()
    else:
        sink.seek(0)
        str_result = sink.read()

    pd_result = exp.to_json(orient="records", lines=lines)

    assert str_result == pd_result

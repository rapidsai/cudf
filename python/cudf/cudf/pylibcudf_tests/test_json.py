# Copyright (c) 2024, NVIDIA CORPORATION.
import io
import os
import pathlib

import pytest

import cudf._lib.pylibcudf as plc


@pytest.mark.parametrize(
    "sink", ["a.txt", pathlib.Path("a.txt"), io.BytesIO(), io.StringIO()]
)
def test_write_json_basic(plc_table_w_meta, sink, tmp_path):
    if isinstance(sink, str):
        sink = f"{tmp_path}/{sink}"
    elif isinstance(sink, os.PathLike):
        sink = tmp_path.joinpath(sink)
    plc.io.json.write_json(
        plc.io.SinkInfo([sink]),
        plc_table_w_meta,
    )

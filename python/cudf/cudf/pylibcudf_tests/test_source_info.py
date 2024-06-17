# Copyright (c) 2024, NVIDIA CORPORATION.

import io

import pyarrow as pa
import pytest

import cudf._lib.pylibcudf as plc
from cudf._lib.pylibcudf.io.datasource import NativeFileDatasource


@pytest.mark.parametrize(
    "source",
    [
        "a.txt",
        b"hello world",
        io.BytesIO(b"hello world"),
        NativeFileDatasource(pa.PythonFile(io.BytesIO(), mode="r")),
    ],
)
def test_source_info_ctor(source, tmp_path):
    if isinstance(source, str):
        file = tmp_path / source
        file.write_bytes("hello world".encode("utf-8"))
        source = str(file)

    plc.io.SourceInfo([source])

    # TODO: test contents of source_info buffer is correct
    # once buffers are exposed on python side


@pytest.mark.parametrize(
    "sources",
    [
        ["a.txt", "a.txt"],
        [b"hello world", b"hello there"],
        [io.BytesIO(b"hello world"), io.BytesIO(b"hello there")],
        [
            NativeFileDatasource(pa.PythonFile(io.BytesIO(), mode="r")),
            NativeFileDatasource(pa.PythonFile(io.BytesIO(), mode="r")),
        ],
    ],
)
def test_source_info_ctor_multiple(sources, tmp_path):
    for i in range(len(sources)):
        source = sources[i]
        if isinstance(source, str):
            file = tmp_path / source
            file.write_bytes("hello world".encode("utf-8"))
            sources[i] = str(file)

    plc.io.SourceInfo(sources)

    # TODO: test contents of source_info buffer is correct
    # once buffers are exposed on python side


@pytest.mark.parametrize(
    "sources",
    [
        ["awef.txt", b"hello world", io.BytesIO(b"hello world")],
        [b"hello world", b"hello there", "awef.txt"],
        [
            io.BytesIO(b"hello world"),
            io.BytesIO(b"hello there"),
            b"hello world",
        ],
        [
            NativeFileDatasource(pa.PythonFile(io.BytesIO(), mode="r")),
            "awef.txt",
            b"hello world",
        ],
    ],
)
def test_source_info_ctor_mixing_invalid(sources, tmp_path):
    # Unlike the previous test
    # don't create files so that they are missing
    for i in range(len(sources)):
        source = sources[i]
        if isinstance(source, str):
            file = tmp_path / source
            file.write_bytes("hello world".encode("utf-8"))
            sources[i] = str(file)
    with pytest.raises(ValueError):
        plc.io.SourceInfo(sources)

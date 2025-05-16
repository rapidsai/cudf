# Copyright (c) 2024, NVIDIA CORPORATION.

import io

import pytest

import pylibcudf as plc


@pytest.fixture(params=[plc.io.SourceInfo, plc.io.SinkInfo])
def io_class(request):
    return request.param


def _skip_invalid_sinks(io_class, sink):
    """
    Skip invalid sinks for SinkInfo
    """
    if io_class is plc.io.SinkInfo and isinstance(sink, bytes):
        pytest.skip("bytes is not a valid input for SinkInfo")


@pytest.mark.parametrize(
    "source",
    [
        "a.txt",
        b"hello world",
        io.BytesIO(b"hello world"),
    ],
)
def test_source_info_ctor(io_class, source, tmp_path):
    if isinstance(source, str):
        file = tmp_path / source
        file.write_bytes("hello world".encode("utf-8"))
        source = str(file)

    _skip_invalid_sinks(io_class, source)

    io_class([source])


@pytest.mark.parametrize(
    "sources",
    [
        ["a.txt"],
        [b"hello world"],
        [io.BytesIO(b"hello world")],
        ["a.txt", "a.txt"],
        [b"hello world", b"hello there"],
        [io.BytesIO(b"hello world"), io.BytesIO(b"hello there")],
    ],
)
def test_source_info_ctor_multiple(io_class, sources, tmp_path):
    for i in range(len(sources)):
        source = sources[i]
        if isinstance(source, str):
            file = tmp_path / source
            file.write_bytes("hello world".encode("utf-8"))
            sources[i] = str(file)

        _skip_invalid_sinks(io_class, source)

    io_class(sources)


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
    ],
)
def test_source_info_ctor_mixing_invalid(io_class, sources, tmp_path):
    # Unlike the previous test
    # don't create files so that they are missing
    for i in range(len(sources)):
        source = sources[i]
        if isinstance(source, str):
            file = tmp_path / source
            file.write_bytes("hello world".encode("utf-8"))
            sources[i] = str(file)
        _skip_invalid_sinks(io_class, source)
    with pytest.raises(ValueError):
        io_class(sources)


def test_source_info_invalid():
    with pytest.raises(ValueError):
        plc.io.SourceInfo([123])

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pylibcudf as plc
from cuda.bindings import runtime

MODE = os.environ.get("GB300_PLUGIN_MODE", "")


def _read_tiny_snappy_parquet():
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pydict({"a": [1, 2, 3]}), buf)
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([buf.getvalue()])
    ).build()
    plc.io.parquet.read_parquet(options)
    (err,) = runtime.cudaDeviceSynchronize()
    assert err == runtime.cudaError_t.cudaSuccess


class _FileBytesSourceInfo(plc.io.SourceInfo):
    def __init__(self, sources):
        if sources and all(
            isinstance(s, (str, os.PathLike))
            and not plc.io.SourceInfo._is_remote_uri(s)
            for s in sources
        ):
            sources = [open(s, "rb").read() for s in sources]
        self._keepalive = sources
        super().__init__(sources)


def pytest_configure(config):
    if MODE == "files-as-bytes":
        plc.io.SourceInfo = _FileBytesSourceInfo
    elif MODE == "warmup":
        _read_tiny_snappy_parquet()
    elif MODE == "sync-before-first-read":
        original = plc.io.parquet.read_parquet
        state = {"first": True}

        def read_parquet(*args, **kwargs):
            if state["first"]:
                state["first"] = False
                (err,) = runtime.cudaDeviceSynchronize()
                assert err == runtime.cudaError_t.cudaSuccess
            return original(*args, **kwargs)

        plc.io.parquet.read_parquet = read_parquet

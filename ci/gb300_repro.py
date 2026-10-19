# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pylibcudf as plc
from rmm.pylibrmm.stream import Stream

writer, stream_kind, iterations = sys.argv[1], sys.argv[2], int(sys.argv[3])

tmp = tempfile.mkdtemp()
read_path = os.path.join(tmp, "read.parquet")
pq.write_table(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}), read_path)
table = plc.Table.from_arrow(
    pa.table({"a": [1.0, 2.0, None], "b": [True, None, False]})
)
persistent_stream = Stream()


def get_stream():
    if stream_kind == "none":
        return None
    if stream_kind == "persistent":
        return persistent_stream
    return Stream()


for _ in range(iterations):
    stream = get_stream()
    if writer == "orc":
        options = (
            plc.io.orc.OrcWriterOptions.builder(
                plc.io.SinkInfo([os.path.join(tmp, "out.orc")]), table
            )
            .compression(plc.io.types.CompressionType.SNAPPY)
            .build()
        )
        plc.io.orc.write_orc(options, stream)
    elif writer == "parquet":
        options = (
            plc.io.parquet.ParquetWriterOptions.builder(
                plc.io.SinkInfo([os.path.join(tmp, "out.parquet")]), table
            )
            .compression(plc.io.types.CompressionType.SNAPPY)
            .build()
        )
        plc.io.parquet.write_parquet(options, stream)
    if stream is not None:
        stream.synchronize()
    del stream

    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([read_path])
    ).build()
    plc.io.parquet.read_parquet(options)

print(f"OK {writer} {stream_kind} {iterations}")

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pylibcudf as plc

mode = sys.argv[1]
iterations = int(sys.argv[2])

with tempfile.TemporaryDirectory() as tmp:
    paths = []
    for i, data in enumerate(
        [{"a": [3, 1, 4], "b": [1, 5, 9]}, {"a": [1, 6], "b": [1, 8]}]
    ):
        path = os.path.join(tmp, f"tbl{i}.parquet")
        pq.write_table(pa.Table.from_pydict(data), path)
        paths.append(path)

    for _ in range(iterations):
        if mode == "path":
            sources = paths
        else:
            sources = [open(p, "rb").read() for p in paths]
        options = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(sources)
        ).build()
        result = plc.io.parquet.read_parquet(options)
        assert result.num_rows_per_source == [3, 2]

print(f"OK {mode} {iterations}")

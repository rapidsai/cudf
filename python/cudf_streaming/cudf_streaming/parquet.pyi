# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.expressions import Expression
from pylibcudf.io.parquet import ParquetReaderOptions

from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.streaming.core.actor import CppActor
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rmm.pylibrmm.stream import Stream

class Filter:
    def __init__(self, stream: Stream, filter: Expression) -> None: ...

def read_parquet(
    ctx: Context,
    comm: Communicator,
    ch_out: Channel[TableChunk],
    num_producers: int,
    options: ParquetReaderOptions,
    num_rows_per_chunk: int,
    filter: Filter | None = None,
) -> CppActor: ...

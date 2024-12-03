# Copyright (c) 2020-2024, NVIDIA CORPORATION.
import numpy as np

from . import (
    binaryop,
    copying,
    csv,
    groupby,
    interop,
    merge,
    nvtext,
    orc,
    parquet,
    reduce,
    round,
    sort,
    stream_compaction,
    string_casting,
    strings,
    strings_udf,
    text,
)

MAX_COLUMN_SIZE = np.iinfo(np.int32).max
MAX_COLUMN_SIZE_STR = "INT32_MAX"
MAX_STRING_COLUMN_BYTES = np.iinfo(np.int32).max
MAX_STRING_COLUMN_BYTES_STR = "INT32_MAX"

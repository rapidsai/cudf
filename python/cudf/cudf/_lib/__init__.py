# Copyright (c) 2020-2024, NVIDIA CORPORATION.
import numpy as np

from . import (
    binaryop,
    copying,
    csv,
    datetime,
    filling,
    groupby,
    interop,
    join,
    json,
    merge,
    null_mask,
    nvtext,
    orc,
    parquet,
    reduce,
    replace,
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

# Copyright (c) 2020-2024, NVIDIA CORPORATION.
import numpy as np

from . import (
    avro,
    binaryop,
    concat,
    copying,
    csv,
    datetime,
    filling,
    groupby,
    hash,
    interop,
    join,
    json,
    labeling,
    merge,
    null_mask,
    nvtext,
    orc,
    parquet,
    partitioning,
    quantiles,
    reduce,
    replace,
    reshape,
    rolling,
    round,
    search,
    sort,
    stream_compaction,
    string_casting,
    strings,
    strings_udf,
    text,
    timezone,
    transpose,
    unary,
)

MAX_COLUMN_SIZE = np.iinfo(np.int32).max
MAX_COLUMN_SIZE_STR = "INT32_MAX"
MAX_STRING_COLUMN_BYTES = np.iinfo(np.int32).max
MAX_STRING_COLUMN_BYTES_STR = "INT32_MAX"

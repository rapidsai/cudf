# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool

from cudf._lib.pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from cudf._lib.pylibcudf.libcudf.io.types cimport compression_type, quote_style
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.pylibcudf.types cimport DataType


cpdef TableWithMetadata read_csv(
    SourceInfo source_info,
    compression_type compression = *,
    size_t byte_range_offset = *,
    size_t byte_range_size = *,
    list col_names = *,
    str prefix = *,
    bool mangle_dupe_cols = *,
    list usecols = *,
    size_type nrows = *,
    size_type skiprows = *,
    size_type skipfooter = *,
    size_type header = *,
    str lineterminator = *,
    str delimiter = *,
    str thousands = *,
    str decimal = *,
    str comment = *,
    bool delim_whitespace = *,
    bool skipinitialspace = *,
    bool skip_blank_lines = *,
    quote_style quoting = *,
    str quotechar = *,
    bool doublequote = *,
    bool detect_whitespace_around_quotes = *,
    list parse_dates = *,
    list parse_hex = *,
    object dtypes = *,
    list true_values = *,
    list false_values = *,
    list na_values = *,
    bool keep_default_na = *,
    bool na_filter = *,
    bool dayfirst = *,
    # Disabled for now, see comments
    # in csv.pyx
    # DataType timestamp_type = *,
)

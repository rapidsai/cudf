# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

import datetime

from pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.orc cimport (
    orc_reader_options,
    read_orc as cpp_read_orc,
)
from pylibcudf.libcudf.io.orc_metadata cimport (
    binary_statistics,
    bucket_statistics,
    column_statistics,
    date_statistics,
    decimal_statistics,
    double_statistics,
    integer_statistics,
    no_statistics,
    read_parsed_orc_statistics as cpp_read_parsed_orc_statistics,
    statistics_type,
    string_statistics,
    timestamp_statistics,
)
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.types cimport DataType
from pylibcudf.variant cimport get_if, holds_alternative


cdef class OrcColumnStatistics:
    def __init__(self):
        raise TypeError(
            "OrcColumnStatistics should not be instantiated by users. If it is "
            "being constructed in Cython from a preexisting libcudf object, "
            "use `OrcColumnStatistics.from_libcudf` instead."
        )

    @property
    def number_of_values(self):
        if self.number_of_values_c.has_value():
            return self.number_of_values_c.value()
        return None

    @property
    def has_null(self):
        if self.has_null_c.has_value():
            return self.has_null_c.value()
        return None

    cdef void _init_stats_dict(self):
        # Initialize stats to return and parse stats blob
        self.column_stats = {}

        cdef statistics_type type_specific_stats = self.type_specific_stats_c

        cdef integer_statistics* int_stats
        cdef double_statistics* dbl_stats
        cdef string_statistics* str_stats
        cdef bucket_statistics* bucket_stats
        cdef decimal_statistics* dec_stats
        cdef date_statistics* date_stats
        cdef binary_statistics* bin_stats
        cdef timestamp_statistics* ts_stats

        if holds_alternative[no_statistics](type_specific_stats):
            pass
        elif int_stats := get_if[integer_statistics](&type_specific_stats):
            if int_stats.minimum.has_value():
                self.column_stats["minimum"] = int_stats.minimum.value()
            else:
                self.column_stats["minimum"] = None
            if int_stats.maximum.has_value():
                self.column_stats["maximum"] = int_stats.maximum.value()
            else:
                self.column_stats["maximum"] = None
            if int_stats.sum.has_value():
                self.column_stats["sum"] = int_stats.sum.value()
            else:
                self.column_stats["sum"] = None
        elif dbl_stats := get_if[double_statistics](&type_specific_stats):
            if dbl_stats.minimum.has_value():
                self.column_stats["minimum"] = dbl_stats.minimum.value()
            else:
                self.column_stats["minimum"] = None
            if dbl_stats.maximum.has_value():
                self.column_stats["maximum"] = dbl_stats.maximum.value()
            else:
                self.column_stats["maximum"] = None
            if dbl_stats.sum.has_value():
                self.column_stats["sum"] = dbl_stats.sum.value()
            else:
                self.column_stats["sum"] = None
        elif str_stats := get_if[string_statistics](&type_specific_stats):
            if str_stats.minimum.has_value():
                self.column_stats["minimum"] = str_stats.minimum.value().decode("utf-8")
            else:
                self.column_stats["minimum"] = None
            if str_stats.maximum.has_value():
                self.column_stats["maximum"] = str_stats.maximum.value().decode("utf-8")
            else:
                self.column_stats["maximum"] = None
            if str_stats.sum.has_value():
                self.column_stats["sum"] = str_stats.sum.value()
            else:
                self.column_stats["sum"] = None
        elif bucket_stats := get_if[bucket_statistics](&type_specific_stats):
            self.column_stats["true_count"] = bucket_stats.count[0]
            self.column_stats["false_count"] = (
                self.number_of_values
                - self.column_stats["true_count"]
            )
        elif dec_stats := get_if[decimal_statistics](&type_specific_stats):
            if dec_stats.minimum.has_value():
                self.column_stats["minimum"] = dec_stats.minimum.value().decode("utf-8")
            else:
                self.column_stats["minimum"] = None
            if dec_stats.maximum.has_value():
                self.column_stats["maximum"] = dec_stats.maximum.value().decode("utf-8")
            else:
                self.column_stats["maximum"] = None
            if dec_stats.sum.has_value():
                self.column_stats["sum"] = dec_stats.sum.value().decode("utf-8")
            else:
                self.column_stats["sum"] = None
        elif date_stats := get_if[date_statistics](&type_specific_stats):
            if date_stats.minimum.has_value():
                self.column_stats["minimum"] = datetime.datetime.fromtimestamp(
                    datetime.timedelta(date_stats.minimum.value()).total_seconds(),
                    datetime.timezone.utc,
                )
            else:
                self.column_stats["minimum"] = None
            if date_stats.maximum.has_value():
                self.column_stats["maximum"] = datetime.datetime.fromtimestamp(
                    datetime.timedelta(date_stats.maximum.value()).total_seconds(),
                    datetime.timezone.utc,
                )
            else:
                self.column_stats["maximum"] = None
        elif bin_stats := get_if[binary_statistics](&type_specific_stats):
            if bin_stats.sum.has_value():
                self.column_stats["sum"] = bin_stats.sum.value()
            else:
                self.column_stats["sum"] = None
        elif ts_stats := get_if[timestamp_statistics](&type_specific_stats):
            # Before ORC-135, the local timezone offset was included and they were
            # stored as minimum and maximum. After ORC-135, the timestamp is
            # adjusted to UTC before being converted to milliseconds and stored
            # in minimumUtc and maximumUtc.
            # TODO: Support minimum and maximum by reading writer's local timezone
            if ts_stats.minimum_utc.has_value() and ts_stats.maximum_utc.has_value():
                self.column_stats["minimum"] = datetime.datetime.fromtimestamp(
                    ts_stats.minimum_utc.value() / 1000, datetime.timezone.utc
                )
                self.column_stats["maximum"] = datetime.datetime.fromtimestamp(
                    ts_stats.maximum_utc.value() / 1000, datetime.timezone.utc
                )
        else:
            raise ValueError("Unsupported statistics type")

    def __getitem__(self, item):
        return self.column_stats[item]

    def __contains__(self, item):
        return item in self.column_stats

    def get(self, item, default=None):
        return self.column_stats.get(item, default)

    @staticmethod
    cdef OrcColumnStatistics from_libcudf(column_statistics& col_stats):
        cdef OrcColumnStatistics out = OrcColumnStatistics.__new__(OrcColumnStatistics)
        out.number_of_values_c = col_stats.number_of_values
        out.has_null_c = col_stats.has_null
        out.type_specific_stats_c = col_stats.type_specific_stats
        out._init_stats_dict()
        return out


cdef class ParsedOrcStatistics:

    @property
    def column_names(self):
        return [name.decode() for name in self.c_obj.column_names]

    @property
    def file_stats(self):
        return [
            OrcColumnStatistics.from_libcudf(self.c_obj.file_stats[i])
            for i in range(self.c_obj.file_stats.size())
        ]

    @property
    def stripes_stats(self):
        return [
            [
                OrcColumnStatistics.from_libcudf(stripe_stats_c[i])
                for i in range(stripe_stats_c.size())
            ]
            for stripe_stats_c in self.c_obj.stripes_stats
        ]

    @staticmethod
    cdef ParsedOrcStatistics from_libcudf(parsed_orc_statistics& orc_stats):
        cdef ParsedOrcStatistics out = ParsedOrcStatistics.__new__(ParsedOrcStatistics)
        out.c_obj = move(orc_stats)
        return out


cpdef TableWithMetadata read_orc(
    SourceInfo source_info,
    list columns = None,
    list stripes = None,
    size_type skip_rows = 0,
    size_type nrows = -1,
    bool use_index = True,
    bool use_np_dtypes = True,
    DataType timestamp_type = None,
    list decimal128_columns = None,
):
    """Reads an ORC file into a :py:class:`~.types.TableWithMetadata`.

    Parameters
    ----------
    source_info : SourceInfo
        The SourceInfo object to read the Parquet file from.
    columns : list, default None
        The string names of the columns to be read.
    stripes : list[list[size_type]], default None
        List of stripes to be read.
    skip_rows : int64_t, default 0
        The number of rows to skip from the start of the file.
    nrows : size_type, default -1
        The number of rows to read. By default, read the entire file.
    use_index : bool, default True
        Whether to use the row index to speed up reading.
    use_np_dtypes : bool, default True
        Whether to use numpy compatible dtypes.
    timestamp_type : DataType, default None
        The timestamp type to use for the timestamp columns.
    decimal128_columns : list, default None
        List of column names to be read as 128-bit decimals.

    Returns
    -------
    TableWithMetadata
        The Table and its corresponding metadata (column names) that were read in.
    """
    cdef orc_reader_options opts
    cdef vector[vector[size_type]] c_stripes
    opts = move(
        orc_reader_options.builder(source_info.c_obj)
        .use_index(use_index)
        .build()
    )
    if nrows >= 0:
        opts.set_num_rows(nrows)
    if skip_rows >= 0:
        opts.set_skip_rows(skip_rows)
    if stripes is not None:
        c_stripes = stripes
        opts.set_stripes(c_stripes)
    if timestamp_type is not None:
        opts.set_timestamp_type(timestamp_type.c_obj)

    cdef vector[string] c_decimal128_columns
    if decimal128_columns is not None and len(decimal128_columns) > 0:
        c_decimal128_columns.reserve(len(decimal128_columns))
        for col in decimal128_columns:
            if not isinstance(col, str):
                raise TypeError("Decimal 128 column names must be strings!")
            c_decimal128_columns.push_back(col.encode())
        opts.set_decimal128_columns(c_decimal128_columns)

    cdef vector[string] c_column_names
    if columns is not None and len(columns) > 0:
        c_column_names.reserve(len(columns))
        for col in columns:
            if not isinstance(col, str):
                raise TypeError("Column names must be strings!")
            c_column_names.push_back(col.encode())
        opts.set_columns(c_column_names)

    cdef table_with_metadata c_result

    with nogil:
        c_result = move(cpp_read_orc(opts))

    return TableWithMetadata.from_libcudf(c_result)


cpdef ParsedOrcStatistics read_parsed_orc_statistics(
    SourceInfo source_info
):
    cdef parsed_orc_statistics parsed = (
        cpp_read_parsed_orc_statistics(source_info.c_obj)
    )
    return ParsedOrcStatistics.from_libcudf(parsed)

# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

import datetime

from cudf._lib.pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from cudf._lib.pylibcudf.libcudf.io.orc cimport (
    orc_reader_options,
    read_orc as cpp_read_orc,
)
from cudf._lib.pylibcudf.libcudf.io.orc_metadata cimport (
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
from cudf._lib.pylibcudf.libcudf.io.types cimport table_with_metadata
from cudf._lib.pylibcudf.libcudf.types cimport size_type, type_id
from cudf._lib.pylibcudf.types cimport DataType
from cudf._lib.pylibcudf.utils.variant cimport get_if, holds_alternative


cdef class OrcColumnStatistics:
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

    def get(self, item, alt=None):
        return self.column_stats.get(item, alt)

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
        stats_lst = []
        for i in range(self.c_obj.file_stats.size()):
            stats_lst.append(OrcColumnStatistics.from_libcudf(self.c_obj.file_stats[i]))
        return stats_lst

    @property
    def stripes_stats(self):
        stats_lst = []
        for stripe_stats_c in self.c_obj.stripes_stats:
            stripe_stats = []
            for i in range(stripe_stats_c.size()):
                stripe_stats.append(OrcColumnStatistics.from_libcudf(stripe_stats_c[i]))
            stats_lst.append(stripe_stats)
        return stats_lst

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
    size_type num_rows = -1,
    bool use_index = True,
    bool use_np_dtypes = True,
    DataType timestamp_type = DataType(type_id.EMPTY),
    list decimal128_columns = None,
):
    """
    """
    cdef orc_reader_options opts
    cdef vector[vector[size_type]] c_stripes
    opts = move(
        orc_reader_options.builder(source_info.c_obj)
        .use_index(use_index)
        .build()
    )
    if num_rows >= 0:
        opts.set_num_rows(num_rows)
    if skip_rows >= 0:
        opts.set_skip_rows(skip_rows)
    if stripes is not None:
        c_stripes = stripes
        opts.set_stripes(c_stripes)
    if timestamp_type.id() != type_id.EMPTY:
        opts.set_timestamp_type(timestamp_type.c_obj)

    cdef vector[string] c_column_names
    if columns is not None:
        c_column_names.reserve(len(columns))
        for col in columns:
            if not isinstance(col, str):
                raise TypeError("Column names must be strings!")
            c_column_names.push_back(str(col).encode())
        if len(columns) > 0:
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

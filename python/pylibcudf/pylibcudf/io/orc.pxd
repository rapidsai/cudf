# Copyright (c) 2024, NVIDIA CORPORATION.
from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.orc_metadata cimport (
    column_statistics,
    parsed_orc_statistics,
    statistics_type,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.types cimport DataType


cpdef TableWithMetadata read_orc(
    SourceInfo source_info,
    list columns = *,
    list stripes = *,
    size_type skip_rows = *,
    size_type nrows = *,
    bool use_index = *,
    bool use_np_dtypes = *,
    DataType timestamp_type = *,
    list decimal128_columns = *
)

cdef class OrcColumnStatistics:
    cdef optional[uint64_t] number_of_values_c
    cdef optional[bool] has_null_c
    cdef statistics_type type_specific_stats_c
    cdef dict column_stats

    cdef void _init_stats_dict(self)

    @staticmethod
    cdef OrcColumnStatistics from_libcudf(column_statistics& col_stats)


cdef class ParsedOrcStatistics:
    cdef parsed_orc_statistics c_obj

    @staticmethod
    cdef ParsedOrcStatistics from_libcudf(parsed_orc_statistics& orc_stats)


cpdef ParsedOrcStatistics read_parsed_orc_statistics(
    SourceInfo source_info
)

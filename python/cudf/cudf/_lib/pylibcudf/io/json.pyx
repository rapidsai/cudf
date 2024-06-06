# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.limits cimport numeric_limits
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.pylibcudf.io.types cimport SinkInfo, TableWithMetadata
from cudf._lib.pylibcudf.libcudf.io.json cimport (
    json_writer_options,
    write_json as cpp_write_json,
)
from cudf._lib.pylibcudf.libcudf.io.types cimport table_metadata
from cudf._lib.pylibcudf.types cimport size_type


cpdef void write_json(
    SinkInfo sink_info,
    TableWithMetadata table_w_meta,
    str na_rep = "",
    bool include_nulls = False,
    bool lines = False,
    int rows_per_chunk = numeric_limits[size_type].max(),
    str true_value = "true",
    str false_value = "false"
):
    """
    """
    cdef table_metadata tbl_meta = table_w_meta.metadata
    cdef string na_rep_c = na_rep.encode()
    cdef string true_value_c = true_value.encode()
    cdef string false_value_c = false_value.encode()

    cdef json_writer_options options = move(
        json_writer_options.builder(sink_info.c_obj, table_w_meta.tbl.view())
        .metadata(tbl_meta)
        .na_rep(na_rep_c)
        .include_nulls(include_nulls)
        .lines(lines)
        .rows_per_chunk(rows_per_chunk)
        .true_value(true_value_c)
        .false_value(false_value_c)
        .build()
    )

    with nogil:
        cpp_write_json(options)

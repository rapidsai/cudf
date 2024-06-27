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
    size_type rows_per_chunk = numeric_limits[size_type].max(),
    str true_value = "true",
    str false_value = "false"
):
    """
    Writes a :py:class:`~cudf._lib.pylibcudf.table.Table` to JSON format.

    Parameters
    ----------
    sink_info: SinkInfo
        The SinkInfo object to write the JSON to.
    table_w_meta: TableWithMetadata
        The TableWithMetadata object containing the Table to write
    na_rep: str, default ""
        The string representation for null values.
    include_nulls: bool, default False
        Enables/Disables output of nulls as 'null'.
    lines: bool, default False
        If `True`, write output in the JSON lines format.
    rows_per_chunk: size_type, defaults to length of the input table
        The maximum number of rows to write at a time.
    true_value: str, default "true"
        The string representation for values != 0 in INT8 types.
    false_value: str, default "false"
        The string representation for values == 0 in INT8 types.
    """
    cdef table_metadata tbl_meta = table_w_meta.metadata
    cdef string na_rep_c = na_rep.encode()

    cdef json_writer_options options = move(
        json_writer_options.builder(sink_info.c_obj, table_w_meta.tbl.view())
        .metadata(tbl_meta)
        .na_rep(na_rep_c)
        .include_nulls(include_nulls)
        .lines(lines)
        .build()
    )

    if rows_per_chunk != numeric_limits[size_type].max():
        options.set_rows_per_chunk(rows_per_chunk)
    if true_value != "true":
        options.set_true_value(<string>true_value.encode())
    if false_value != "false":
        options.set_false_value(<string>false_value.encode())

    with nogil:
        cpp_write_json(options)

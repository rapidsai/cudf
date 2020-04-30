# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.io.functions cimport (
    write_csv as cpp_write_csv,
    write_csv_args
)

from cudf._lib.cpp.io.types cimport (
    sink_info,
    table_metadata
)

from cudf._lib.table cimport Table
from cudf._lib.cpp.table.table_view cimport table_view

from libcpp cimport bool
from libcpp.string cimport string

cpdef write_csv(
    #Table table,
    #filepath,
    #object sep,
    #object na_rep,
    #bool header,
    #object line_terminator,
    #int rows_per_chunk,
    Table table,
    file_path=None,
    sep=",",
    na_rep="",
    header=True,
    line_terminator="\n",
    rows_per_chunk=8,

):
    """
    Cython function to call into libcudf API, see `write_csv`.

    See Also
    --------
    cudf.io.csv.write_csv
    """

    cdef table_view input_table_view = table.data_view()
    cdef bool include_header_c = header
    cdef char delim_c = ord(sep)
    cdef string line_term_c = line_terminator.encode()
    cdef string na_c = na_rep.encode()
    cdef int rows_per_hunk_c = rows_per_chunk
    cdef table_metadata metadata_ = table_metadata()
    cdef string true_value_c = 'True'.encode()
    cdef string false_value_c = 'False'.encode()
    cdef sink_info snk = sink_info(<string>str(file_path).encode())

    if table._column_names is not None:
        metadata_.column_names.reserve(len(table._column_names))
        for col_name in table._column_names:
            metadata_.column_names.push_back(str.encode(col_name))

    cdef write_csv_args write_csv_args_c = write_csv_args(snk, input_table_view, na_c, include_header_c, rows_per_hunk_c, line_term_c, delim_c, true_value_c, false_value_c, &metadata_)

    with nogil:
        cpp_write_csv(write_csv_args_c)

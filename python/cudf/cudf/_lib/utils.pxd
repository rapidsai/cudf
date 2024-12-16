# Copyright (c) 2020-2024, NVIDIA CORPORATION.

cpdef data_from_pylibcudf_table(tbl, column_names, index_names=*)
cpdef data_from_pylibcudf_io(tbl_with_meta, column_names = *, index_names = *)
cpdef columns_from_pylibcudf_table(tbl)
cpdef _data_from_columns(columns, column_names, index_names=*)

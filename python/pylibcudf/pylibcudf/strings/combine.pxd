# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.strings.combine cimport (
    output_if_empty_list,
    separator_on_nulls,
)
from pylibcudf.scalar cimport Scalar
from pylibcudf.table cimport Table

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column concatenate(
    Table strings_columns,
    ColumnOrScalar separator,
    Scalar narep=*,
    Scalar col_narep=*,
    separator_on_nulls separate_nulls=*,
)

cpdef Column join_strings(Column input, Scalar separator, Scalar narep)


cpdef Column join_list_elements(
    Column source_strings,
    ColumnOrScalar separator,
    Scalar separator_narep,
    Scalar string_narep,
    separator_on_nulls separate_nulls,
    output_if_empty_list empty_list_policy,
)

# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import pylibcudf as plc

from cudf._lib.column cimport Column


def transpose(list source_columns):
    """Transpose m n-row columns into n m-row columns
    """
    input_table = plc.table.Table(
        [col.to_pylibcudf(mode="read") for col in source_columns]
    )
    _, result_table = plc.transpose.transpose(input_table)
    return [Column.from_pylibcudf(col) for col in result_table.columns()]

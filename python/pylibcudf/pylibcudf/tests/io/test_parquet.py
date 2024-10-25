# Copyright (c) 2024, NVIDIA CORPORATION.
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from pyarrow.parquet import read_table
from utils import assert_table_and_meta_eq, make_source

import pylibcudf as plc
from pylibcudf.expressions import (
    ASTOperator,
    ColumnNameReference,
    ColumnReference,
    Literal,
    Operation,
)

# Shared kwargs to pass to make_source
_COMMON_PARQUET_SOURCE_KWARGS = {"format": "parquet"}


@pytest.mark.parametrize("columns", [None, ["col_int64", "col_bool"]])
def test_read_parquet_basic(
    table_data, binary_source_or_sink, nrows_skiprows, columns
):
    _, pa_table = table_data
    nrows, skiprows = nrows_skiprows

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )

    res = plc.io.parquet.read_parquet(
        plc.io.SourceInfo([source]),
        nrows=nrows,
        skip_rows=skiprows,
        columns=columns,
    )

    if columns is not None:
        pa_table = pa_table.select(columns)

    # Adapt to nrows/skiprows
    pa_table = pa_table.slice(
        offset=skiprows, length=nrows if nrows != -1 else None
    )

    assert_table_and_meta_eq(pa_table, res, check_field_nullability=False)


@pytest.mark.parametrize(
    "pa_filters,plc_filters",
    [
        (
            pc.field("col_int64") >= 10,
            Operation(
                ASTOperator.GREATER_EQUAL,
                ColumnNameReference("col_int64"),
                Literal(plc.interop.from_arrow(pa.scalar(10))),
            ),
        ),
        (
            (pc.field("col_int64") >= 10) & (pc.field("col_double") < 0),
            Operation(
                ASTOperator.LOGICAL_AND,
                Operation(
                    ASTOperator.GREATER_EQUAL,
                    ColumnNameReference("col_int64"),
                    Literal(plc.interop.from_arrow(pa.scalar(10))),
                ),
                Operation(
                    ASTOperator.LESS,
                    ColumnNameReference("col_double"),
                    Literal(plc.interop.from_arrow(pa.scalar(0.0))),
                ),
            ),
        ),
        (
            (pc.field(0) == 10),
            Operation(
                ASTOperator.EQUAL,
                ColumnReference(0),
                Literal(plc.interop.from_arrow(pa.scalar(10))),
            ),
        ),
    ],
)
def test_read_parquet_filters(
    table_data, binary_source_or_sink, pa_filters, plc_filters
):
    _, pa_table = table_data

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )

    plc_table_w_meta = plc.io.parquet.read_parquet(
        plc.io.SourceInfo([source]), filters=plc_filters
    )
    exp = read_table(source, filters=pa_filters)
    assert_table_and_meta_eq(
        exp, plc_table_w_meta, check_field_nullability=False
    )


# TODO: Test these options
# list row_groups = None,
# ^^^ This one is not tested since it's not in pyarrow/pandas, deprecate?
# bool convert_strings_to_categories = False,
# bool use_pandas_metadata = True

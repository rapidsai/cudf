# Copyright (c) 2024, NVIDIA CORPORATION.
import pyarrow as pa
import pytest
from utils import _convert_types, assert_table_and_meta_eq, make_source

import pylibcudf as plc

# Shared kwargs to pass to make_source
_COMMON_ORC_SOURCE_KWARGS = {"format": "orc"}


@pytest.mark.parametrize("columns", [None, ["col_int64", "col_bool"]])
def test_read_orc_basic(
    table_data, binary_source_or_sink, nrows_skiprows, columns
):
    _, pa_table = table_data
    nrows, skiprows = nrows_skiprows

    # ORC reader doesn't support skip_rows for nested columns
    if skiprows > 0:
        colnames_to_drop = []
        for i in range(len(pa_table.schema)):
            field = pa_table.schema.field(i)

            if pa.types.is_nested(field.type):
                colnames_to_drop.append(field.name)
        pa_table = pa_table.drop(colnames_to_drop)
    # ORC doesn't support unsigned ints
    # let's cast to int64
    _, new_fields = _convert_types(
        pa_table, pa.types.is_unsigned_integer, pa.int64()
    )
    pa_table = pa_table.cast(pa.schema(new_fields))

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_ORC_SOURCE_KWARGS
    )

    res = plc.io.orc.read_orc(
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

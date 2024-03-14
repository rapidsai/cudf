# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa

from cudf._lib import pylibcudf as plc


# TODO: The two functions below should probably be a function in interop. We should move
# the class method on Table into a function in interop, and then use that to implement
# this function in interop as well.
def column_from_arrow(pa_array):
    """Create a pylibcudf column from a PyArrow array."""
    pa_table = pa.table([pa_array], [""])
    table = plc.Table.from_arrow(pa_table)
    return table.columns()[0]


def column_to_arrow(plc_column):
    """Create a PyArrow array from a pylibcudf column."""
    return plc.Table([plc_column]).to_arrow([plc.interop.ColumnMetadata("")])[
        0
    ]


# This function is just a convenient wrapper for the metadata handling. We should update
# the API to make naming more convenient when we don't need metadata objects.
def table_to_arrow(plc_table):
    """Create a PyArrow table from a pylibcudf Table."""
    return plc_table.to_arrow(
        [plc.interop.ColumnMetadata("")] * plc_table.num_columns()
    )


def scalar_to_arrow(plc_scalar):
    return plc_scalar.to_arrow(plc.interop.ColumnMetadata(""))


def assert_array_eq(plc_column, pa_array):
    """Verify that the pylibcudf array and PyArrow array are equal."""
    pa_equal = pa.compute.equal(column_to_arrow(plc_column), pa_array)
    assert pa.compute.all(pa_equal).as_py()


def assert_table_eq(plc_table, pa_table):
    """Verify that the pylibcudf array and PyArrow array are equal."""
    plc_shape = (plc_table.num_rows(), plc_table.num_columns())
    assert plc_shape == pa_table.shape

    for plc_col, pa_col in zip(plc_table.columns(), pa_table.columns):
        assert_array_eq(plc_col, pa_col)

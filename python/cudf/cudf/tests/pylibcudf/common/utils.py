# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa

from cudf._lib import pylibcudf as plc


def assert_array_eq(plc_column, pa_array):
    """Verify that the pylibcudf array and PyArrow array are equal."""
    plc_as_pa = plc.Table([plc_column]).to_arrow(
        [plc.interop.ColumnMetadata("tmp")]
    )[0]
    pa_equal = pa.compute.equal(plc_as_pa, pa_array)
    assert pa.compute.all(pa_equal).as_py()


def column_from_arrow(pa_array):
    """Create a pylibcudf column from a PyArrow array."""
    # TODO: This should probably be a function in interop. We should move the class
    # method on Table into a function in interop, and then use that to implement this
    # function in inteorp as well.

    pa_tbl = pa.table([pa_array], ["tmp"])
    tbl = plc.Table.from_arrow(pa_tbl)
    return tbl.columns()[0]

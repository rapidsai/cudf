# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa


def assert_array_eq(plc_array, pa_array):
    """Verify that the pylibcudf array and PyArrow array are equal."""
    assert pa.compute.all(
        pa.compute.equal(plc_array.to_arrow(), pa_array)
    ).as_py()

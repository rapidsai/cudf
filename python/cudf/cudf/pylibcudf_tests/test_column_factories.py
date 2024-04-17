# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc


def test_make_empty_column(pa_type):
    pa_col = pa.array([], type=pa_type)

    # TODO: DataType.from_arrow()?
    plc_type = plc.interop.from_arrow(pa_col).type()

    if isinstance(pa_type, (pa.ListType, pa.StructType)):
        with pytest.raises(ValueError):
            plc.column_factories.make_empty_column(plc_type)
        return

    cudf_col = plc.column_factories.make_empty_column(plc_type)
    assert_column_eq(cudf_col, pa_col)

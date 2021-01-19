# Copyright (c) 2021, NVIDIA CORPORATION.

from decimal import Decimal

import pyarrow as pa
import pytest

from cudf.core.column import DecimalColumn


@pytest.mark.parametrize(
    "data",
    [
        [Decimal("1.1"), Decimal("2.2"), Decimal("3.3"), Decimal("4.4")],
        [Decimal("-1.1"), Decimal("2.2"), Decimal("3.3"), Decimal("4.4")],
        [1],
        [-1],
        [1, 2, 3, 4],
        [42, 1729, 4104],
        [1, 2, None, 4],
        [None, None, None],
        [],
    ],
)
@pytest.mark.parametrize(
    "typ",
    [
        pa.decimal128(precision=4, scale=2),
        pa.decimal128(precision=5, scale=3),
        pa.decimal128(precision=6, scale=4),
    ],
)
def test_round_trip_decimal_column(data, typ):
    pa_arr = pa.array(data, type=typ)
    col = DecimalColumn.from_arrow(pa_arr)
    assert pa_arr.equals(col.to_arrow())


def test_from_arrow_max_precision():
    with pytest.raises(ValueError):
        DecimalColumn.from_arrow(
            pa.array([1, 2, 3], type=pa.decimal128(scale=0, precision=19))
        )

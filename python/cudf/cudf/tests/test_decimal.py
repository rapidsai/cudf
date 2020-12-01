import pyarrow as pa

from cudf.core.column import DecimalColumn


@pytest.mark.parametrize(
    "data", [[1.1, 2.2, 3.3, 4.4], [1, 2, 3, 4], [42, 1729, 4104]]
)
@pytest.mark.parametrize(
    "typ",
    [
        pa.decimal128(precision=3, scale=1),
        pa.decimal128(precision=4, scale=2),
        pa.decimal128(precision=5, scale=3),
        pa.decimal128(precision=6, scale=4),
    ],
)
def test_round_trip_decimal_column(data, typ):
    pa_arr = pa.array(data, type=typ)
    col = DecimalColumn.from_arrow(pa_arr)
    assert pa_arr.equals(col.to_arrow())

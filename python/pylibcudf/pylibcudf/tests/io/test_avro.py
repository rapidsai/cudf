# Copyright (c) 2024, NVIDIA CORPORATION.

import io
import itertools

import fastavro
import pyarrow as pa
import pytest
from utils import assert_table_and_meta_eq

import pylibcudf as plc

avro_dtype_pairs = [
    ("boolean", pa.bool_()),
    ("int", pa.int32()),
    ("long", pa.int64()),
    ("float", pa.float32()),
    ("double", pa.float64()),
    ("bytes", pa.string()),
    ("string", pa.string()),
]


@pytest.fixture(
    scope="module", params=itertools.combinations(avro_dtype_pairs, 2)
)
def avro_dtypes(request):
    return request.param


@pytest.fixture
def avro_dtype_data(avro_dtypes):
    (avro_type1, _), (avro_type2, _) = avro_dtypes

    def _get_data(avro_type):
        if avro_type == "boolean":
            return [True, False, True]
        elif avro_type in {"int", "long"}:
            return [1, 2, -1]
        elif avro_type in {"float", "double"}:
            return [1.0, 3.1415, -3.1415]
        elif avro_type == "bytes":
            return [b"a", b"b", b"c"]
        elif avro_type == "string":
            return ["Hello", "World!", ""]

    return _get_data(avro_type1), _get_data(avro_type2)


@pytest.fixture(
    params=[
        (0, 0),
        (0, -1),
        (1, -1),
        (3, -1),
    ]
)
def row_opts(request):
    """
    (skip_rows, num_rows) combos for the avro reader
    """
    return request.param


@pytest.mark.parametrize("columns", [["prop1"], [], ["prop1", "prop2"]])
@pytest.mark.parametrize("nullable", [True, False])
def test_read_avro(avro_dtypes, avro_dtype_data, row_opts, columns, nullable):
    (avro_type1, expected_type1), (avro_type2, expected_type2) = avro_dtypes

    avro_type1 = avro_type1 if not nullable else ["null", avro_type1]
    avro_type2 = avro_type2 if not nullable else ["null", avro_type2]

    skip_rows, num_rows = row_opts

    schema = fastavro.parse_schema(
        {
            "type": "record",
            "name": "test",
            "fields": [
                {"name": "prop1", "type": avro_type1},
                {"name": "prop2", "type": avro_type2},
            ],
        }
    )

    if nullable:
        avro_dtype_data = (
            avro_dtype_data[0] + [None],
            avro_dtype_data[1] + [None],
        )

    records = [
        {"prop1": val1, "prop2": val2} for val1, val2 in zip(*avro_dtype_data)
    ]

    buffer = io.BytesIO()
    fastavro.writer(buffer, schema, records)
    buffer.seek(0)

    res = plc.io.avro.read_avro(
        plc.io.types.SourceInfo([buffer]),
        columns=columns,
        skip_rows=skip_rows,
        num_rows=num_rows,
    )

    expected = pa.Table.from_arrays(
        [
            pa.array(avro_dtype_data[0], type=expected_type1),
            pa.array(avro_dtype_data[1], type=expected_type2),
        ],
        names=["prop1", "prop2"],
    )

    # Adjust for skip_rows/num_rows in result
    length = num_rows if num_rows != -1 else None
    expected = expected.slice(skip_rows, length=length)

    # adjust for # of columns
    if columns != []:
        expected = expected.select(columns)

    assert_table_and_meta_eq(expected, res)

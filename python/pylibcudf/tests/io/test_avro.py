# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import io
import itertools

import fastavro
import pyarrow as pa
import pytest
from utils import assert_table_and_meta_eq

from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.stream import Stream

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


def _make_avro_table(avro_dtypes, avro_dtype_data, nullable=False):
    (avro_type1, expected_type1), (avro_type2, expected_type2) = avro_dtypes

    avro_type1 = avro_type1 if not nullable else ["null", avro_type1]
    avro_type2 = avro_type2 if not nullable else ["null", avro_type2]

    if nullable:
        avro_dtype_data = (
            avro_dtype_data[0] + [None],
            avro_dtype_data[1] + [None],
        )

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

    records = [
        {"prop1": val1, "prop2": val2}
        for val1, val2 in zip(*avro_dtype_data, strict=True)
    ]

    buffer = io.BytesIO()
    fastavro.writer(buffer, schema, records)
    buffer.seek(0)

    expected = pa.Table.from_arrays(
        [
            pa.array(avro_dtype_data[0], type=expected_type1),
            pa.array(avro_dtype_data[1], type=expected_type2),
        ],
        names=["prop1", "prop2"],
    )

    return buffer, expected


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize("columns", [["prop1"], [], ["prop1", "prop2"]])
@pytest.mark.parametrize("nullable", [True, False])
@pytest.mark.parametrize("source_strategy", ["inline", "set_source"])
def test_read_avro(
    avro_dtypes,
    avro_dtype_data,
    row_opts,
    columns,
    nullable,
    stream,
    source_strategy,
):
    skip_rows, num_rows = row_opts
    buffer, expected = _make_avro_table(avro_dtypes, avro_dtype_data, nullable)

    source = plc.io.types.SourceInfo([buffer])
    builder = plc.io.avro.AvroReaderOptions.builder(
        source if source_strategy == "inline" else plc.io.types.SourceInfo([])
    )
    options = (
        builder.columns(columns)
        .skip_rows(skip_rows)
        .num_rows(num_rows)
        .build()
    )

    if source_strategy == "set_source":
        options.set_source(source)

    res = plc.io.avro.read_avro(options, stream)

    length = num_rows if num_rows != -1 else None
    expected = expected.slice(skip_rows, length=length)

    if columns != []:
        expected = expected.select(columns)

    assert_table_and_meta_eq(expected, res)


@pytest.mark.parametrize("stream", [None, Stream()])
def test_read_avro_from_device_buffers(avro_dtypes, avro_dtype_data, stream):
    buffer, expected = _make_avro_table(avro_dtypes, avro_dtype_data)
    buf = buffer.getbuffer()
    device_buf = DeviceBuffer.to_device(buf, plc.utils._get_stream(stream))

    options = plc.io.avro.AvroReaderOptions.builder(
        plc.io.types.SourceInfo([device_buf])
    ).build()

    res = plc.io.avro.read_avro(options, stream)

    expected = pa.concat_tables([expected])
    assert_table_and_meta_eq(expected, res)

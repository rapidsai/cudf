# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import io
from typing import Optional

import fastavro
import pytest

import cudf
from cudf.testing._utils import assert_eq
from cudf.testing.dataset_generator import rand_dataframe


def cudf_from_avro_util(schema: dict, records: list) -> cudf.DataFrame:
    schema = [] if schema is None else fastavro.parse_schema(schema)
    buffer = io.BytesIO()
    fastavro.writer(buffer, schema, records)
    buffer.seek(0)
    return cudf.read_avro(buffer)


avro_type_params = [
    ("boolean", "bool"),
    ("int", "int32"),
    ("long", "int64"),
    ("float", "float32"),
    ("double", "float64"),
    ("bytes", "str"),
    ("string", "str"),
]


@pytest.mark.parametrize("avro_type, expected_dtype", avro_type_params)
@pytest.mark.parametrize("namespace", [None, "root_ns"])
@pytest.mark.parametrize("nullable", [True, False])
def test_can_detect_dtype_from_avro_type(
    avro_type, expected_dtype, namespace, nullable
):
    avro_type = avro_type if not nullable else ["null", avro_type]

    schema = fastavro.parse_schema(
        {
            "type": "record",
            "name": "test",
            "namespace": namespace,
            "fields": [{"name": "prop", "type": avro_type}],
        }
    )

    actual = cudf_from_avro_util(schema, [])

    expected = cudf.DataFrame(
        {"prop": cudf.Series(None, None, expected_dtype)}
    )

    assert_eq(expected, actual)


@pytest.mark.parametrize("avro_type, expected_dtype", avro_type_params)
@pytest.mark.parametrize("namespace", [None, "root_ns"])
@pytest.mark.parametrize("nullable", [True, False])
def test_can_detect_dtype_from_avro_type_nested(
    avro_type, expected_dtype, namespace, nullable
):
    avro_type = avro_type if not nullable else ["null", avro_type]

    schema_leaf = {
        "name": "leaf",
        "type": "record",
        "fields": [{"name": "prop3", "type": avro_type}],
    }

    schema_child = {
        "name": "child",
        "type": "record",
        "fields": [{"name": "prop2", "type": schema_leaf}],
    }

    schema_root = {
        "name": "root",
        "type": "record",
        "namespace": namespace,
        "fields": [{"name": "prop1", "type": schema_child}],
    }

    actual = cudf_from_avro_util(schema_root, [])

    col_name = "{ns}child.{ns}leaf.prop3".format(
        ns="" if namespace is None else namespace + "."
    )

    expected = cudf.DataFrame(
        {col_name: cudf.Series(None, None, expected_dtype)}
    )

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "avro_type, cudf_type, avro_val, cudf_val",
    [
        ("boolean", "bool", True, True),
        ("boolean", "bool", False, False),
        ("int", "int32", 1234, 1234),
        ("long", "int64", 1234, 1234),
        ("float", "float32", 12.34, 12.34),
        ("double", "float64", 12.34, 12.34),
        ("string", "str", "heyϴ", "heyϴ"),
        # ("bytes", "str", "heyϴ", "heyϴ"),
    ],
)
def test_can_parse_single_value(avro_type, cudf_type, avro_val, cudf_val):

    schema_root = {
        "name": "root",
        "type": "record",
        "fields": [{"name": "prop", "type": ["null", avro_type]}],
    }

    records = [
        {"prop": avro_val},
    ]

    actual = cudf_from_avro_util(schema_root, records)

    expected = cudf.DataFrame(
        {"prop": cudf.Series(data=[cudf_val], dtype=cudf_type)}
    )

    assert_eq(expected, actual)


@pytest.mark.parametrize("avro_type, cudf_type", avro_type_params)
def test_can_parse_single_null(avro_type, cudf_type):

    schema_root = {
        "name": "root",
        "type": "record",
        "fields": [{"name": "prop", "type": ["null", avro_type]}],
    }

    records = [{"prop": None}]

    actual = cudf_from_avro_util(schema_root, records)

    expected = cudf.DataFrame(
        {"prop": cudf.Series(data=[None], dtype=cudf_type)}
    )

    assert_eq(expected, actual)


@pytest.mark.parametrize("avro_type, cudf_type", avro_type_params)
def test_can_parse_no_data(avro_type, cudf_type):

    schema_root = {
        "name": "root",
        "type": "record",
        "fields": [{"name": "prop", "type": ["null", avro_type]}],
    }

    records = []

    actual = cudf_from_avro_util(schema_root, records)

    expected = cudf.DataFrame({"prop": cudf.Series(data=[], dtype=cudf_type)})

    assert_eq(expected, actual)


@pytest.mark.xfail(
    reason="cudf avro reader is unable to parse zero-field metadata."
)
@pytest.mark.parametrize("avro_type, cudf_type", avro_type_params)
def test_can_parse_no_fields(avro_type, cudf_type):

    schema_root = {
        "name": "root",
        "type": "record",
        "fields": [],
    }

    records = []

    actual = cudf_from_avro_util(schema_root, records)

    expected = cudf.DataFrame()

    assert_eq(expected, actual)


def test_can_parse_no_schema():

    schema_root = None
    records = []
    actual = cudf_from_avro_util(schema_root, records)
    expected = cudf.DataFrame()
    assert_eq(expected, actual)


@pytest.mark.parametrize("rows", [0, 1, 10, 1000])
@pytest.mark.parametrize("codec", ["null", "deflate", "snappy"])
def test_avro_compression(rows, codec):
    schema = {
        "name": "root",
        "type": "record",
        "fields": [
            {"name": "0", "type": "int"},
            {"name": "1", "type": "string"},
        ],
    }

    df = rand_dataframe(
        [
            {"dtype": "int32", "null_frequency": 0, "cardinality": 1000},
            {
                "dtype": "str",
                "null_frequency": 0,
                "cardinality": 100,
                "max_string_length": 10,
            },
        ],
        rows,
    )
    expected_df = cudf.DataFrame.from_arrow(df)

    records = df.to_pandas().to_dict(orient="records")

    buffer = io.BytesIO()
    fastavro.writer(buffer, schema, records, codec=codec)
    buffer.seek(0)
    got_df = cudf.read_avro(buffer)

    assert_eq(expected_df, got_df)


avro_logical_type_params = [
    # (avro logical type, avro primitive type, cudf expected dtype)
    ("date", "int", "datetime64[s]"),
]


@pytest.mark.parametrize(
    "logical_type, primitive_type, expected_dtype", avro_logical_type_params
)
@pytest.mark.parametrize("namespace", [None, "root_ns"])
@pytest.mark.parametrize("nullable", [True, False])
@pytest.mark.parametrize("prepend_null", [True, False])
def test_can_detect_dtypes_from_avro_logical_type(
    logical_type,
    primitive_type,
    expected_dtype,
    namespace,
    nullable,
    prepend_null,
):
    avro_type = [{"logicalType": logical_type, "type": primitive_type}]
    if nullable:
        if prepend_null:
            avro_type.insert(0, "null")
        else:
            avro_type.append("null")

    schema = fastavro.parse_schema(
        {
            "type": "record",
            "name": "test",
            "namespace": namespace,
            "fields": [{"name": "prop", "type": avro_type}],
        }
    )

    actual = cudf_from_avro_util(schema, [])

    expected = cudf.DataFrame(
        {"prop": cudf.Series(None, None, expected_dtype)}
    )

    assert_eq(expected, actual)


def get_days_from_epoch(date: Optional[datetime.date]) -> Optional[int]:
    if date is None:
        return None
    return (date - datetime.date(1970, 1, 1)).days


@pytest.mark.parametrize("namespace", [None, "root_ns"])
@pytest.mark.parametrize("nullable", [True, False])
@pytest.mark.parametrize("prepend_null", [True, False])
def test_can_parse_avro_date_logical_type(namespace, nullable, prepend_null):

    avro_type = {"logicalType": "date", "type": "int"}
    if nullable:
        if prepend_null:
            avro_type = ["null", avro_type]
        else:
            avro_type = [avro_type, "null"]

    schema_dict = {
        "type": "record",
        "name": "test",
        "fields": [
            {"name": "o_date", "type": avro_type},
        ],
    }

    if namespace:
        schema_dict["namespace"] = namespace

    schema = fastavro.parse_schema(schema_dict)

    if nullable:
        # Insert some None values in no particular order.  These will get
        # converted into avro "nulls" by the fastavro writer.
        dates = [
            None,
            datetime.date(1970, 1, 1),
            datetime.date(1970, 1, 2),
            datetime.date(1981, 10, 25),
            None,
            None,
            datetime.date(2012, 5, 18),
            None,
            datetime.date(2019, 9, 3),
            None,
        ]
    else:
        dates = [
            datetime.date(1970, 1, 1),
            datetime.date(1970, 1, 2),
            datetime.date(1981, 10, 25),
            datetime.date(2012, 5, 18),
            datetime.date(2019, 9, 3),
        ]

    days_from_epoch = [get_days_from_epoch(date) for date in dates]

    records = [{"o_date": day} for day in days_from_epoch]

    actual = cudf_from_avro_util(schema, records)

    expected = cudf.DataFrame(
        {"o_date": cudf.Series(dates, dtype="datetime64[s]")}
    )

    assert_eq(expected, actual)

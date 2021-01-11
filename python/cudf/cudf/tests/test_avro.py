# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cudf.tests.utils import assert_eq
import io
import cudf
import fastavro
import pytest

def cudf_from_avro_util(schema, records):
    schema = fastavro.parse_schema(schema)
    buffer = io.BytesIO()
    fastavro.writer(buffer, schema, records)
    buffer.seek(0)
    return cudf.read_avro(buffer)

avro_type_params = [
    # ('null', 'int64'), # seems to be ignored?
    ('boolean', 'bool'),
    ('int', 'int32'),
    ('long', 'int64'),
    ('float', 'float32'),
    ('double', 'float64'),
    ('bytes', 'str'),
    ('string', 'str'),
]

@pytest.mark.parametrize("avro_type, expected_dtype", avro_type_params)
def test_can_detect_dtype(avro_type, expected_dtype):
    schema = fastavro.parse_schema({
        'type': 'record',
        'name': 'test',
        'fields': [ {'name': 'x', 'type': avro_type } ],
    })

    actual = cudf_from_avro_util(schema, [])

    expected = cudf.DataFrame({
        'x': cudf.Series(None, None, expected_dtype)
    })

    assert_eq(expected, actual)

@pytest.mark.parametrize("avro_type, expected_dtype", avro_type_params)
@pytest.mark.parametrize("namespace", [None, 'root_ns'])
def test_can_detect_dtype_nested(avro_type, expected_dtype, namespace):
    
    schema_leaf = {
        'name': 'leaf',
        'type': 'record',
        'fields': [ { 'name': 'prop3', 'type': avro_type } ]
    }

    schema_child = {
        'name': 'child',
        'type': 'record',
        'fields': [ { 'name': 'prop2', 'type': schema_leaf } ]
    }

    schema_parent = {
        'name': 'root',
        'type': 'record',
        'namespace': namespace,
        'fields': [ { 'name': 'prop1', 'type': schema_child } ],
    }

    actual = cudf_from_avro_util(schema_parent, [])

    col_name = "{ns}child.{ns}leaf.prop3".format(ns='' if namespace is None else namespace + '.')

    expected = cudf.DataFrame({
        col_name: cudf.Series(None, None, expected_dtype)
    })

    assert_eq(expected, actual)

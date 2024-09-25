/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/json.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <string>

// Base test fixture for tests
struct JsonWSNormalizationTest : public cudf::test::BaseFixture {};

TEST_F(JsonWSNormalizationTest, ReadJsonOption)
{
  // When mixed type fields are read as strings, the table read will differ depending the
  // value of normalize_whitespace

  // Test input
  std::string const host_input = "{ \"a\" : {\"b\" :\t\"c\"}}";
  cudf::io::json_reader_options input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{host_input.data(), host_input.size()})
      .lines(true)
      .mixed_types_as_string(true)
      .normalize_whitespace(true);

  cudf::io::table_with_metadata processed_table = cudf::io::read_json(input_options);

  // Expected table
  std::string const expected_input = R"({ "a" : {"b":"c"}})";
  cudf::io::json_reader_options expected_input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{expected_input.data(), expected_input.size()})
      .lines(true)
      .mixed_types_as_string(true)
      .normalize_whitespace(false);

  cudf::io::table_with_metadata expected_table = cudf::io::read_json(expected_input_options);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table.tbl->view(), processed_table.tbl->view());
}

TEST_F(JsonWSNormalizationTest, ReadJsonOption_InvalidRows)
{
  // When mixed type fields are read as strings, the table read will differ depending the
  // value of normalize_whitespace

  // Test input
  std::string const host_input = R"(
  { "Root": { "Key": [ { "EE": tr ue } ] } }
  { "Root": { "Key": "abc" } }
  { "Root": { "Key": [ { "EE": 12 34 } ] } }
  { "Root": { "Key": [{ "YY": 1}] } }
  { "Root": { "Key": [ { "EE": 12. 34 } ] } }
  { "Root": { "Key": [ { "EE": "efg" } ] } }
  )";
  cudf::io::json_reader_options input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{host_input.data(), host_input.size()})
      .lines(true)
      .mixed_types_as_string(true)
      .normalize_whitespace(true)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);

  cudf::io::table_with_metadata processed_table = cudf::io::read_json(input_options);

  // Expected table
  std::string const expected_input = R"(
  { "Root": { "Key": [ { "EE": tr ue } ] } }
  { "Root": { "Key": "abc" } }
  { "Root": { "Key": [ { "EE": 12 34 } ] } }
  { "Root": { "Key": [{"YY":1}] } }
  { "Root": { "Key": [ { "EE": 12. 34 } ] } }
  { "Root": { "Key": [{"EE":"efg"}] } }
  )";
  cudf::io::json_reader_options expected_input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{expected_input.data(), expected_input.size()})
      .lines(true)
      .mixed_types_as_string(true)
      .normalize_whitespace(false)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);

  cudf::io::table_with_metadata expected_table = cudf::io::read_json(expected_input_options);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table.tbl->view(), processed_table.tbl->view());
}

TEST_F(JsonWSNormalizationTest, ReadJsonOption_InvalidRows_NoMixedType)
{
  // When mixed type fields are read as strings, the table read will differ depending the
  // value of normalize_whitespace

  // Test input
  std::string const host_input = R"(
  { "Root": { "Key": [ { "EE": tr ue } ] } }
  { "Root": { "Key": [ { "EE": 12 34 } ] } }
  { "Root": { "Key": [{ "YY": 1}] } }
  { "Root": { "Key": [ { "EE": 12. 34 } ] } }
  { "Root": { "Key": [ { "EE": "efg" }, { "YY" :   "abc" }    ] } }
  { "Root": { "Key": [  { "YY" :   "abc" }    ] } }
  )";

  std::map<std::string, cudf::io::schema_element> dtype_schema{
    {"Key", {cudf::data_type{cudf::type_id::STRING}}}};

  cudf::io::json_reader_options input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{host_input.data(), host_input.size()})
      .dtypes(dtype_schema)
      .lines(true)
      .prune_columns(true)
      .normalize_whitespace(true)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);

  cudf::io::table_with_metadata processed_table = cudf::io::read_json(input_options);

  // Expected table
  std::string const expected_input = R"(
  { "Root": { "Key": [ { "EE": tr ue } , { "YY" :    2 } ] } }
  { "Root": { "Key": [ { "EE": 12 34 } ] } }
  { "Root": { "Key": [{"YY":1}] } }
  { "Root": { "Key": [ { "EE": 12. 34 } ] } }
  { "Root": { "Key": [{"EE":"efg"},{"YY":"abc"}] } }
  { "Root": { "Key": [{"YY":"abc"}] } }
  )";

  cudf::io::json_reader_options expected_input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{expected_input.data(), expected_input.size()})
      .dtypes(dtype_schema)
      .lines(true)
      .prune_columns(true)
      .normalize_whitespace(false)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);

  cudf::io::table_with_metadata expected_table = cudf::io::read_json(expected_input_options);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table.tbl->view(), processed_table.tbl->view());
}

CUDF_TEST_PROGRAM_MAIN()

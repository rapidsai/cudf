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

void run_test(std::string const& host_input, std::string const& expected_host_output)
{
  // Prepare cuda stream for data transfers & kernels
  auto stream_view = cudf::test::get_default_stream();

  auto device_input = rmm::device_buffer(
    host_input.c_str(), host_input.size(), stream_view, cudf::get_current_device_resource_ref());

  // Preprocessing FST
  cudf::io::datasource::owning_buffer<rmm::device_buffer> device_data(std::move(device_input));
  cudf::io::json::detail::normalize_whitespace(
    device_data, stream_view, cudf::get_current_device_resource_ref());

  std::string preprocessed_host_output(device_data.size(), 0);
  CUDF_CUDA_TRY(cudaMemcpyAsync(preprocessed_host_output.data(),
                                device_data.data(),
                                preprocessed_host_output.size(),
                                cudaMemcpyDeviceToHost,
                                stream_view.value()));

  stream_view.synchronize();
  ASSERT_EQ(preprocessed_host_output.size(), expected_host_output.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(
    preprocessed_host_output, expected_host_output, preprocessed_host_output.size());
}

TEST_F(JsonWSNormalizationTest, GroundTruth_Spaces)
{
  std::string input  = R"({ "A" : "TEST" })";
  std::string output = R"({"A":"TEST"})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_MoreSpaces)
{
  std::string input  = R"({"a": [1, 2, 3, 4, 5, 6, 7, 8], "b": {"c": "d"}})";
  std::string output = R"({"a":[1,2,3,4,5,6,7,8],"b":{"c":"d"}})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_SpacesInString)
{
  std::string input  = R"({" a ":50})";
  std::string output = R"({" a ":50})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_NewlineInString)
{
  std::string input  = "{\"a\" : \"x\ny\"}\n{\"a\" : \"x\\ny\"}";
  std::string output = "{\"a\":\"x\ny\"}\n{\"a\":\"x\\ny\"}";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_Tabs)
{
  std::string input  = "{\"a\":\t\"b\"}";
  std::string output = R"({"a":"b"})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_SpacesAndTabs)
{
  std::string input  = "{\"A\" : \t\"TEST\" }";
  std::string output = R"({"A":"TEST"})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_MultilineJSONWithSpacesAndTabs)
{
  std::string input =
    "{ \"foo rapids\": [1,2,3], \"bar\trapids\": 123 }\n\t{ \"foo rapids\": { \"a\": 1 }, "
    "\"bar\trapids\": 456 }";
  std::string output =
    "{\"foo rapids\":[1,2,3],\"bar\trapids\":123}\n{\"foo rapids\":{\"a\":1},\"bar\trapids\":456}";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_PureJSONExample)
{
  std::string input  = R"([{"a":50}, {"a" : 60}])";
  std::string output = R"([{"a":50},{"a":60}])";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_NoNormalizationRequired)
{
  std::string input  = R"({"a\\n\r\a":50})";
  std::string output = R"({"a\\n\r\a":50})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_InvalidInput)
{
  std::string input  = "{\"a\" : \"b }\n{ \"c \" :\t\"d\"}";
  std::string output = "{\"a\":\"b }\n{\"c \":\"d\"}";
  run_test(input, output);
}

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

CUDF_TEST_PROGRAM_MAIN()

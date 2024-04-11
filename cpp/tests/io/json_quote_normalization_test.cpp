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
#include <cudf_test/default_stream.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/detail/json.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <string>

// Base test fixture for tests
struct JsonNormalizationTest : public cudf::test::BaseFixture {};

void run_test(const std::string& host_input, const std::string& expected_host_output)
{
  // RMM memory resource
  std::shared_ptr<rmm::mr::device_memory_resource> rsc =
    std::make_shared<rmm::mr::cuda_memory_resource>();

  rmm::device_uvector<char> device_input(
    host_input.size(), cudf::test::get_default_stream(), rsc.get());
  CUDF_CUDA_TRY(cudaMemcpyAsync(device_input.data(),
                                host_input.data(),
                                host_input.size(),
                                cudaMemcpyHostToDevice,
                                cudf::test::get_default_stream().value()));
  // Preprocessing FST
  auto device_fst_output = cudf::io::json::detail::normalize_single_quotes(
    std::move(device_input), cudf::test::get_default_stream(), rsc.get());

  std::string preprocessed_host_output(device_fst_output.size(), 0);
  CUDF_CUDA_TRY(cudaMemcpyAsync(preprocessed_host_output.data(),
                                device_fst_output.data(),
                                preprocessed_host_output.size(),
                                cudaMemcpyDeviceToHost,
                                cudf::test::get_default_stream().value()));
  CUDF_TEST_EXPECT_VECTOR_EQUAL(
    preprocessed_host_output, expected_host_output, preprocessed_host_output.size());
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Single)
{
  std::string input  = R"({'A':"TEST'"} ['OTHER STUFF'])";
  std::string output = R"({"A":"TEST'"} ["OTHER STUFF"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_MoreSingle)
{
  std::string input  = R"(['\t','\\t','\\','\\\"\'\\\\','\n','\b','\u0012'])";
  std::string output = R"(["\t","\\t","\\","\\\"'\\\\","\n","\b","\u0012"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_DoubleInSingle)
{
  std::string input  = R"({"A":'TEST"'})";
  std::string output = R"({"A":"TEST\""})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_MoreDoubleInSingle)
{
  std::string input = R"({"ain't ain't a word and you ain't supposed to say it":'"""""""""""'})";
  std::string output =
    R"({"ain't ain't a word and you ain't supposed to say it":"\"\"\"\"\"\"\"\"\"\"\""})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_StillMoreDoubleInSingle)
{
  std::string input  = R"([{"ABC':'CBA":'XYZ":"ZXY'}])";
  std::string output = R"([{"ABC':'CBA":"XYZ\":\"ZXY"}])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_DoubleInSingleAndViceVersa)
{
  std::string input  = R"(['{"A": "B"}',"{'A': 'B'}"])";
  std::string output = R"(["{\"A\": \"B\"}","{'A': 'B'}"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_DoubleAndSingleInSingle)
{
  std::string input  = R"({"\"'\"'\"'\"'":'"\'"\'"\'"\'"'})";
  std::string output = R"({"\"'\"'\"'\"'":"\"'\"'\"'\"'\""})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_EscapedSingleInDouble)
{
  std::string input  = R"(["\t","\\t","\\","\\\'\"\\\\","\n","\b"])";
  std::string output = R"(["\t","\\t","\\","\\'\"\\\\","\n","\b"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_EscapedDoubleInSingle)
{
  std::string input  = R"(["\t","\\t","\\",'\\\'\"\\\\',"\n","\b"])";
  std::string output = R"(["\t","\\t","\\","\\'\"\\\\","\n","\b"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_MismatchedQuotes)
{
  std::string input  = R"(["THIS IS A TEST'])";
  std::string output = R"(["THIS IS A TEST'])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_MismatchedQuotesEscapedOutput)
{
  std::string input  = R"(['THIS IS A TEST"])";
  std::string output = R"(["THIS IS A TEST\"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_MoreMismatchedQuotes)
{
  std::string input  = R"({"MORE TEST'N":'RESUL})";
  std::string output = R"({"MORE TEST'N":"RESUL})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_NoEndQuote)
{
  std::string input  = R"({"NUMBER":100'0,'STRING':'SOMETHING'})";
  std::string output = R"({"NUMBER":100"0,"STRING":"SOMETHING"})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_InvalidJSON)
{
  std::string input  = R"({'NUMBER':100"0,"STRING":"SOMETHING"})";
  std::string output = R"({"NUMBER":100"0,"STRING":"SOMETHING"})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_WrongBackslash)
{
  std::string input  = R"({'a':'\\''})";
  std::string output = R"({"a":"\\""})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_WrongBraces)
{
  std::string input  = R"(}'a': 'b'{)";
  std::string output = R"(}"a": "b"{)";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, ReadJsonOption)
{
  // RMM memory resource
  std::shared_ptr<rmm::mr::device_memory_resource> rsc =
    std::make_shared<rmm::mr::cuda_memory_resource>();

  // Test input
  std::string const host_input = R"({"A":'TEST"'})";
  cudf::io::json_reader_options input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{host_input.data(), host_input.size()})
      .lines(true)
      .normalize_single_quotes(true);

  cudf::io::table_with_metadata processed_table =
    cudf::io::read_json(input_options, cudf::test::get_default_stream(), rsc.get());

  // Expected table
  std::string const expected_input = R"({"A":"TEST\""})";
  cudf::io::json_reader_options expected_input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{expected_input.data(), expected_input.size()})
      .lines(true);

  cudf::io::table_with_metadata expected_table =
    cudf::io::read_json(expected_input_options, cudf::test::get_default_stream(), rsc.get());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table.tbl->view(), processed_table.tbl->view());
}

TEST_F(JsonNormalizationTest, ErrorCheck)
{
  // RMM memory resource
  std::shared_ptr<rmm::mr::device_memory_resource> rsc =
    std::make_shared<rmm::mr::cuda_memory_resource>();

  // Test input
  std::string const host_input = R"({"A":'TEST"'})";
  cudf::io::json_reader_options input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{host_input.data(), host_input.size()})
      .lines(true);

  EXPECT_THROW(cudf::io::read_json(input_options, cudf::test::get_default_stream(), rsc.get()),
               cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()

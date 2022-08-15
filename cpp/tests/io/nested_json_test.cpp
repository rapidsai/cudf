/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <io/json/nested_json.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/lists/lists_column_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <string>

namespace cuio_json = cudf::io::json;

namespace {
// Forward declaration
void print_column(std::string const& input,
                  cuio_json::json_column const& column,
                  uint32_t indent = 0);

/**
 * @brief Helper to generate indentation
 */
std::string pad(uint32_t indent = 0)
{
  std::string pad{};
  if (indent > 0) pad.insert(pad.begin(), indent, ' ');
  return pad;
}

/**
 * @brief Prints a string column.
 */
void print_json_string_col(std::string const& input,
                           cuio_json::json_column const& column,
                           uint32_t indent = 0)
{
  for (std::size_t i = 0; i < column.string_offsets.size(); i++) {
    std::cout << pad(indent) << i << ": [" << (column.validity[i] ? "1" : "0") << "] '"
              << input.substr(column.string_offsets[i], column.string_lengths[i]) << "'\n";
  }
}

/**
 * @brief Prints a list column.
 */
void print_json_list_col(std::string const& input,
                         cuio_json::json_column const& column,
                         uint32_t indent = 0)
{
  std::cout << pad(indent) << " [LIST]\n";
  std::cout << pad(indent) << " -> num. child-columns: " << column.child_columns.size() << "\n";
  std::cout << pad(indent) << " -> num. rows: " << column.current_offset << "\n";
  std::cout << pad(indent) << " -> num. valid: " << column.valid_count << "\n";
  std::cout << pad(indent) << " offsets[]: "
            << "\n";
  for (std::size_t i = 0; i < column.child_offsets.size() - 1; i++) {
    std::cout << pad(indent + 2) << i << ": [" << (column.validity[i] ? "1" : "0") << "] ["
              << column.child_offsets[i] << ", " << column.child_offsets[i + 1] << ")\n";
  }
  if (column.child_columns.size() > 0) {
    std::cout << pad(indent) << column.child_columns.begin()->first << "[]: "
              << "\n";
    print_column(input, column.child_columns.begin()->second, indent + 2);
  }
}

/**
 * @brief Prints a struct column.
 */
void print_json_struct_col(std::string const& input,
                           cuio_json::json_column const& column,
                           uint32_t indent = 0)
{
  std::cout << pad(indent) << " [STRUCT]\n";
  std::cout << pad(indent) << " -> num. child-columns: " << column.child_columns.size() << "\n";
  std::cout << pad(indent) << " -> num. rows: " << column.current_offset << "\n";
  std::cout << pad(indent) << " -> num. valid: " << column.valid_count << "\n";
  std::cout << pad(indent) << " -> validity[]: "
            << "\n";
  for (decltype(column.current_offset) i = 0; i < column.current_offset; i++) {
    std::cout << pad(indent + 2) << i << ": [" << (column.validity[i] ? "1" : "0") << "]\n";
  }
  auto it = std::begin(column.child_columns);
  for (std::size_t i = 0; i < column.child_columns.size(); i++) {
    std::cout << pad(indent + 2) << "child #" << i << " '" << it->first << "'[] \n";
    print_column(input, it->second, indent + 2);
    it++;
  }
}

/**
 * @brief Prints the column's data and recurses through and prints all the child columns.
 */
void print_column(std::string const& input, cuio_json::json_column const& column, uint32_t indent)
{
  switch (column.type) {
    case cuio_json::json_col_t::StringColumn: print_json_string_col(input, column, indent); break;
    case cuio_json::json_col_t::ListColumn: print_json_list_col(input, column, indent); break;
    case cuio_json::json_col_t::StructColumn: print_json_struct_col(input, column, indent); break;
    case cuio_json::json_col_t::Unknown: std::cout << pad(indent) << "[UNKNOWN]\n"; break;
    default: break;
  }
}
}  // namespace

// Base test fixture for tests
struct JsonTest : public cudf::test::BaseFixture {
};

TEST_F(JsonTest, StackContext)
{
  // Type used to represent the atomic symbol type used within the finite-state machine
  using SymbolT      = char;
  using StackSymbolT = char;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  // Test input
  std::string input = R"(  [{)"
                      R"("category": "reference",)"
                      R"("index:": [4,12,42],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "[Sayings of the Century]",)"
                      R"("price": 8.95)"
                      R"(},  )"
                      R"({)"
                      R"("category": "reference",)"
                      R"("index": [4,{},null,{"a":[{ }, {}] } ],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "{}\\\"[], <=semantic-symbols-string\\\\",)"
                      R"("price": 8.95)"
                      R"(}] )";

  // Prepare input & output buffers
  rmm::device_uvector<SymbolT> d_input(input.size(), stream_view);
  hostdevice_vector<StackSymbolT> stack_context(input.size(), stream_view);

  ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_input.data(),
                                        input.data(),
                                        input.size() * sizeof(SymbolT),
                                        cudaMemcpyHostToDevice,
                                        stream.value()));

  // Run algorithm
  cuio_json::detail::get_stack_context(d_input, stack_context.device_ptr(), stream_view);

  // Copy back the results
  stack_context.device_to_host(stream_view);

  // Make sure we copied back the stack context
  stream_view.synchronize();

  std::vector<char> golden_stack_context{
    '_', '_', '_', '[', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '[', '[', '[', '[', '[', '[', '[', '[', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '[', '[', '[', '[', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '[', '[', '[', '{', '[', '[', '[', '[', '[', '[', '[', '{',
    '{', '{', '{', '{', '[', '{', '{', '[', '[', '[', '{', '[', '{', '{', '[', '[', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '[', '_'};

  ASSERT_EQ(golden_stack_context.size(), stack_context.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(golden_stack_context, stack_context, stack_context.size());
}

TEST_F(JsonTest, StackContextUtf8)
{
  // Type used to represent the atomic symbol type used within the finite-state machine
  using SymbolT      = char;
  using StackSymbolT = char;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  // Test input
  std::string input = R"([{"a":{"year":1882,"author": "Bharathi"}, {"a":"filip ʒakotɛ"}}])";

  // Prepare input & output buffers
  rmm::device_uvector<SymbolT> d_input(input.size(), stream_view);
  hostdevice_vector<StackSymbolT> stack_context(input.size(), stream_view);

  ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_input.data(),
                                        input.data(),
                                        input.size() * sizeof(SymbolT),
                                        cudaMemcpyHostToDevice,
                                        stream.value()));

  // Run algorithm
  cuio_json::detail::get_stack_context(d_input, stack_context.device_ptr(), stream_view);

  // Copy back the results
  stack_context.device_to_host(stream_view);

  // Make sure we copied back the stack context
  stream_view.synchronize();

  std::vector<char> golden_stack_context{
    '_', '[', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '['};

  ASSERT_EQ(golden_stack_context.size(), stack_context.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(golden_stack_context, stack_context, stack_context.size());
}

TEST_F(JsonTest, TokenStream)
{
  using cuio_json::PdaTokenT;
  using cuio_json::SymbolOffsetT;
  using cuio_json::SymbolT;

  constexpr std::size_t single_item = 1;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  // Test input
  std::string input = R"(  [{)"
                      R"("category": "reference",)"
                      R"("index:": [4,12,42],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "[Sayings of the Century]",)"
                      R"("price": 8.95)"
                      R"(},  )"
                      R"({)"
                      R"("category": "reference",)"
                      R"("index": [4,{},null,{"a":[{ }, {}] } ],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "{}[], <=semantic-symbols-string",)"
                      R"("price": 8.95)"
                      R"(}] )";

  // Prepare input & output buffers
  rmm::device_uvector<SymbolT> d_input(input.size(), stream_view);

  ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_input.data(),
                                        input.data(),
                                        input.size() * sizeof(SymbolT),
                                        cudaMemcpyHostToDevice,
                                        stream.value()));

  hostdevice_vector<PdaTokenT> tokens_gpu{input.size(), stream_view};
  hostdevice_vector<SymbolOffsetT> token_indices_gpu{input.size(), stream_view};
  hostdevice_vector<SymbolOffsetT> num_tokens_out{single_item, stream_view};

  // Parse the JSON and get the token stream
  cuio_json::detail::get_token_stream(d_input,
                                      tokens_gpu.device_ptr(),
                                      token_indices_gpu.device_ptr(),
                                      num_tokens_out.device_ptr(),
                                      stream_view);

  // Copy back the number of tokens that were written
  num_tokens_out.device_to_host(stream_view);
  tokens_gpu.device_to_host(stream_view);
  token_indices_gpu.device_to_host(stream_view);

  // Make sure we copied back all relevant data
  stream_view.synchronize();

  // Golden token stream sample
  using token_t = cuio_json::token_t;
  std::vector<std::pair<std::size_t, cuio_json::PdaTokenT>> golden_token_stream = {
    {2, token_t::ListBegin},        {3, token_t::StructBegin},      {4, token_t::FieldNameBegin},
    {13, token_t::FieldNameEnd},    {16, token_t::StringBegin},     {26, token_t::StringEnd},
    {28, token_t::FieldNameBegin},  {35, token_t::FieldNameEnd},    {38, token_t::ListBegin},
    {39, token_t::ValueBegin},      {40, token_t::ValueEnd},        {41, token_t::ValueBegin},
    {43, token_t::ValueEnd},        {44, token_t::ValueBegin},      {46, token_t::ValueEnd},
    {46, token_t::ListEnd},         {48, token_t::FieldNameBegin},  {55, token_t::FieldNameEnd},
    {58, token_t::StringBegin},     {69, token_t::StringEnd},       {71, token_t::FieldNameBegin},
    {77, token_t::FieldNameEnd},    {80, token_t::StringBegin},     {105, token_t::StringEnd},
    {107, token_t::FieldNameBegin}, {113, token_t::FieldNameEnd},   {116, token_t::ValueBegin},
    {120, token_t::ValueEnd},       {120, token_t::StructEnd},      {124, token_t::StructBegin},
    {125, token_t::FieldNameBegin}, {134, token_t::FieldNameEnd},   {137, token_t::StringBegin},
    {147, token_t::StringEnd},      {149, token_t::FieldNameBegin}, {155, token_t::FieldNameEnd},
    {158, token_t::ListBegin},      {159, token_t::ValueBegin},     {160, token_t::ValueEnd},
    {161, token_t::StructBegin},    {162, token_t::StructEnd},      {164, token_t::ValueBegin},
    {168, token_t::ValueEnd},       {169, token_t::StructBegin},    {170, token_t::FieldNameBegin},
    {172, token_t::FieldNameEnd},   {174, token_t::ListBegin},      {175, token_t::StructBegin},
    {177, token_t::StructEnd},      {180, token_t::StructBegin},    {181, token_t::StructEnd},
    {182, token_t::ListEnd},        {184, token_t::StructEnd},      {186, token_t::ListEnd},
    {188, token_t::FieldNameBegin}, {195, token_t::FieldNameEnd},   {198, token_t::StringBegin},
    {209, token_t::StringEnd},      {211, token_t::FieldNameBegin}, {217, token_t::FieldNameEnd},
    {220, token_t::StringBegin},    {252, token_t::StringEnd},      {254, token_t::FieldNameBegin},
    {260, token_t::FieldNameEnd},   {263, token_t::ValueBegin},     {267, token_t::ValueEnd},
    {267, token_t::StructEnd},      {268, token_t::ListEnd}};

  // Verify the number of tokens matches
  ASSERT_EQ(golden_token_stream.size(), num_tokens_out[0]);

  for (std::size_t i = 0; i < num_tokens_out[0]; i++) {
    // Ensure the index the tokens are pointing to do match
    EXPECT_EQ(golden_token_stream[i].first, token_indices_gpu[i]) << "Mismatch at #" << i;

    // Ensure the token category is correct
    EXPECT_EQ(golden_token_stream[i].second, tokens_gpu[i]) << "Mismatch at #" << i;
  }
}

TEST_F(JsonTest, ExtractColumn)
{
  using cuio_json::SymbolT;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  std::string input = R"( [{"a":0.0, "b":1.0}, {"a":0.1, "b":1.1}, {"a":0.2, "b":1.2}] )";
  // Get the JSON's tree representation
  auto const cudf_table = cuio_json::detail::parse_nested_json(
    cudf::host_span<SymbolT const>{input.data(), input.size()}, stream_view);

  auto const expected_col_count  = 2;
  auto const first_column_index  = 0;
  auto const second_column_index = 1;
  EXPECT_EQ(cudf_table.tbl->num_columns(), expected_col_count);

  auto expected_col1            = cudf::test::strings_column_wrapper({"0.0", "0.1", "0.2"});
  auto expected_col2            = cudf::test::strings_column_wrapper({"1.0", "1.1", "1.2"});
  cudf::column_view parsed_col1 = cudf_table.tbl->get_column(first_column_index);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col1, parsed_col1);
  cudf::column_view parsed_col2 = cudf_table.tbl->get_column(second_column_index);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col2, parsed_col2);
}

TEST_F(JsonTest, UTF_JSON)
{
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  // Only ASCII string
  std::string ascii_pass = R"([
  {"a":1,"b":2,"c":[3], "d": {}},
  {"a":1,"b":4.0,"c":[], "d": {"year":1882,"author": "Bharathi"}},
  {"a":1,"b":6.0,"c":[5, 7], "d": null},
  {"a":1,"b":8.0,"c":null, "d": {}},
  {"a":1,"b":null,"c":null},
  {"a":1,"b":Infinity,"c":[null], "d": {"year":-600,"author": "Kaniyan"}}])";

  CUDF_EXPECT_NO_THROW(cuio_json::detail::parse_nested_json(ascii_pass, stream_view));

  // utf-8 string that fails parsing.
  std::string utf_failed = R"([
  {"a":1,"b":2,"c":[3], "d": {}},
  {"a":1,"b":4.0,"c":[], "d": {"year":1882,"author": "Bharathi"}},
  {"a":1,"b":6.0,"c":[5, 7], "d": null},
  {"a":1,"b":8.0,"c":null, "d": {}},
  {"a":1,"b":null,"c":null},
  {"a":1,"b":Infinity,"c":[null], "d": {"year":-600,"author": "filip ʒakotɛ"}}])";
  CUDF_EXPECT_NO_THROW(cuio_json::detail::parse_nested_json(utf_failed, stream_view));

  // utf-8 string that passes parsing.
  std::string utf_pass = R"([
  {"a":1,"b":2,"c":[3], "d": {}},
  {"a":1,"b":4.0,"c":[], "d": {"year":1882,"author": "Bharathi"}},
  {"a":1,"b":6.0,"c":[5, 7], "d": null},
  {"a":1,"b":8.0,"c":null, "d": {}},
  {"a":1,"b":null,"c":null},
  {"a":1,"b":Infinity,"c":[null], "d": {"year":-600,"author": "Kaniyan"}},
  {"a":1,"b":NaN,"c":[null, null], "d": {"year": 2, "author": "filip ʒakotɛ"}}])";
  CUDF_EXPECT_NO_THROW(cuio_json::detail::parse_nested_json(utf_pass, stream_view));
}

TEST_F(JsonTest, FromParquet)
{
  using cuio_json::SymbolT;

  std::string input =
    R"([{"0":{},"1":[],"2":{}},{"1":[[""],[]],"2":{"2":""}},{"0":{"a":"1"},"2":{"0":"W&RR=+I","1":""}}])";

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  // Binary parquet data containing the same data as the data represented by the JSON string.
  // We could add a dataset to include this file, but we don't want tests in cudf to have data.
  const unsigned char parquet_data[] = {
    0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0x18, 0x15, 0x18, 0x2C, 0x15, 0x06, 0x15, 0x00, 0x15,
    0x06, 0x15, 0x06, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x21, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x31, 0x15, 0x00, 0x15, 0x24, 0x15, 0x20, 0x2C, 0x15, 0x08, 0x15, 0x00, 0x15, 0x06, 0x15, 0x06,
    0x00, 0x00, 0x12, 0x18, 0x03, 0x00, 0x00, 0x00, 0x03, 0x10, 0x00, 0x05, 0x07, 0x04, 0x2D, 0x00,
    0x01, 0x01, 0x15, 0x00, 0x15, 0x22, 0x15, 0x22, 0x2C, 0x15, 0x06, 0x15, 0x00, 0x15, 0x06, 0x15,
    0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x04, 0x07, 0x00, 0x00, 0x00, 0x57, 0x26, 0x52,
    0x52, 0x3D, 0x2B, 0x49, 0x15, 0x00, 0x15, 0x14, 0x15, 0x14, 0x2C, 0x15, 0x06, 0x15, 0x00, 0x15,
    0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00, 0x15,
    0x00, 0x15, 0x14, 0x15, 0x14, 0x2C, 0x15, 0x06, 0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x03, 0x02, 0x00, 0x00, 0x00, 0x00, 0x15, 0x02, 0x19, 0xCC, 0x48, 0x06,
    0x73, 0x63, 0x68, 0x65, 0x6D, 0x61, 0x15, 0x06, 0x00, 0x35, 0x02, 0x18, 0x01, 0x30, 0x15, 0x02,
    0x00, 0x15, 0x0C, 0x25, 0x02, 0x18, 0x01, 0x61, 0x25, 0x00, 0x00, 0x35, 0x02, 0x18, 0x01, 0x31,
    0x15, 0x02, 0x15, 0x06, 0x00, 0x35, 0x04, 0x18, 0x04, 0x6C, 0x69, 0x73, 0x74, 0x15, 0x02, 0x00,
    0x35, 0x00, 0x18, 0x07, 0x65, 0x6C, 0x65, 0x6D, 0x65, 0x6E, 0x74, 0x15, 0x02, 0x15, 0x06, 0x00,
    0x35, 0x04, 0x18, 0x04, 0x6C, 0x69, 0x73, 0x74, 0x15, 0x02, 0x00, 0x15, 0x0C, 0x25, 0x00, 0x18,
    0x07, 0x65, 0x6C, 0x65, 0x6D, 0x65, 0x6E, 0x74, 0x25, 0x00, 0x00, 0x35, 0x00, 0x18, 0x01, 0x32,
    0x15, 0x06, 0x00, 0x15, 0x0C, 0x25, 0x02, 0x18, 0x01, 0x30, 0x25, 0x00, 0x00, 0x15, 0x0C, 0x25,
    0x02, 0x18, 0x01, 0x31, 0x25, 0x00, 0x00, 0x15, 0x0C, 0x25, 0x02, 0x18, 0x01, 0x32, 0x25, 0x00,
    0x00, 0x16, 0x06, 0x19, 0x1C, 0x19, 0x5C, 0x26, 0x00, 0x1C, 0x15, 0x0C, 0x19, 0x25, 0x00, 0x06,
    0x19, 0x28, 0x01, 0x30, 0x01, 0x61, 0x15, 0x00, 0x16, 0x06, 0x16, 0x3A, 0x16, 0x3A, 0x26, 0x08,
    0x3C, 0x36, 0x04, 0x28, 0x01, 0x31, 0x18, 0x01, 0x31, 0x00, 0x00, 0x00, 0x26, 0x00, 0x1C, 0x15,
    0x0C, 0x19, 0x25, 0x00, 0x06, 0x19, 0x58, 0x01, 0x31, 0x04, 0x6C, 0x69, 0x73, 0x74, 0x07, 0x65,
    0x6C, 0x65, 0x6D, 0x65, 0x6E, 0x74, 0x04, 0x6C, 0x69, 0x73, 0x74, 0x07, 0x65, 0x6C, 0x65, 0x6D,
    0x65, 0x6E, 0x74, 0x15, 0x02, 0x16, 0x08, 0x16, 0x46, 0x16, 0x42, 0x26, 0x42, 0x3C, 0x36, 0x00,
    0x28, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x26, 0x00, 0x1C, 0x15, 0x0C, 0x19, 0x25, 0x00, 0x06,
    0x19, 0x28, 0x01, 0x32, 0x01, 0x30, 0x15, 0x00, 0x16, 0x06, 0x16, 0x44, 0x16, 0x44, 0x26, 0x84,
    0x01, 0x3C, 0x36, 0x04, 0x28, 0x07, 0x57, 0x26, 0x52, 0x52, 0x3D, 0x2B, 0x49, 0x18, 0x07, 0x57,
    0x26, 0x52, 0x52, 0x3D, 0x2B, 0x49, 0x00, 0x00, 0x00, 0x26, 0x00, 0x1C, 0x15, 0x0C, 0x19, 0x25,
    0x00, 0x06, 0x19, 0x28, 0x01, 0x32, 0x01, 0x31, 0x15, 0x00, 0x16, 0x06, 0x16, 0x36, 0x16, 0x36,
    0x26, 0xC8, 0x01, 0x3C, 0x36, 0x04, 0x28, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x26, 0x00, 0x1C,
    0x15, 0x0C, 0x19, 0x25, 0x00, 0x06, 0x19, 0x28, 0x01, 0x32, 0x01, 0x32, 0x15, 0x00, 0x16, 0x06,
    0x16, 0x36, 0x16, 0x36, 0x26, 0xFE, 0x01, 0x3C, 0x36, 0x04, 0x28, 0x00, 0x18, 0x00, 0x00, 0x00,
    0x00, 0x16, 0xAC, 0x02, 0x16, 0x06, 0x00, 0x19, 0x1C, 0x18, 0x06, 0x70, 0x61, 0x6E, 0x64, 0x61,
    0x73, 0x18, 0xFE, 0x04, 0x7B, 0x22, 0x69, 0x6E, 0x64, 0x65, 0x78, 0x5F, 0x63, 0x6F, 0x6C, 0x75,
    0x6D, 0x6E, 0x73, 0x22, 0x3A, 0x20, 0x5B, 0x7B, 0x22, 0x6B, 0x69, 0x6E, 0x64, 0x22, 0x3A, 0x20,
    0x22, 0x72, 0x61, 0x6E, 0x67, 0x65, 0x22, 0x2C, 0x20, 0x22, 0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A,
    0x20, 0x6E, 0x75, 0x6C, 0x6C, 0x2C, 0x20, 0x22, 0x73, 0x74, 0x61, 0x72, 0x74, 0x22, 0x3A, 0x20,
    0x30, 0x2C, 0x20, 0x22, 0x73, 0x74, 0x6F, 0x70, 0x22, 0x3A, 0x20, 0x33, 0x2C, 0x20, 0x22, 0x73,
    0x74, 0x65, 0x70, 0x22, 0x3A, 0x20, 0x31, 0x7D, 0x5D, 0x2C, 0x20, 0x22, 0x63, 0x6F, 0x6C, 0x75,
    0x6D, 0x6E, 0x5F, 0x69, 0x6E, 0x64, 0x65, 0x78, 0x65, 0x73, 0x22, 0x3A, 0x20, 0x5B, 0x7B, 0x22,
    0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A, 0x20, 0x6E, 0x75, 0x6C, 0x6C, 0x2C, 0x20, 0x22, 0x66, 0x69,
    0x65, 0x6C, 0x64, 0x5F, 0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A, 0x20, 0x6E, 0x75, 0x6C, 0x6C, 0x2C,
    0x20, 0x22, 0x70, 0x61, 0x6E, 0x64, 0x61, 0x73, 0x5F, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x20,
    0x22, 0x75, 0x6E, 0x69, 0x63, 0x6F, 0x64, 0x65, 0x22, 0x2C, 0x20, 0x22, 0x6E, 0x75, 0x6D, 0x70,
    0x79, 0x5F, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x20, 0x22, 0x6F, 0x62, 0x6A, 0x65, 0x63, 0x74,
    0x22, 0x2C, 0x20, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22, 0x3A, 0x20, 0x7B,
    0x22, 0x65, 0x6E, 0x63, 0x6F, 0x64, 0x69, 0x6E, 0x67, 0x22, 0x3A, 0x20, 0x22, 0x55, 0x54, 0x46,
    0x2D, 0x38, 0x22, 0x7D, 0x7D, 0x5D, 0x2C, 0x20, 0x22, 0x63, 0x6F, 0x6C, 0x75, 0x6D, 0x6E, 0x73,
    0x22, 0x3A, 0x20, 0x5B, 0x7B, 0x22, 0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A, 0x20, 0x22, 0x30, 0x22,
    0x2C, 0x20, 0x22, 0x66, 0x69, 0x65, 0x6C, 0x64, 0x5F, 0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A, 0x20,
    0x22, 0x30, 0x22, 0x2C, 0x20, 0x22, 0x70, 0x61, 0x6E, 0x64, 0x61, 0x73, 0x5F, 0x74, 0x79, 0x70,
    0x65, 0x22, 0x3A, 0x20, 0x22, 0x6F, 0x62, 0x6A, 0x65, 0x63, 0x74, 0x22, 0x2C, 0x20, 0x22, 0x6E,
    0x75, 0x6D, 0x70, 0x79, 0x5F, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x20, 0x22, 0x6F, 0x62, 0x6A,
    0x65, 0x63, 0x74, 0x22, 0x2C, 0x20, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22,
    0x3A, 0x20, 0x6E, 0x75, 0x6C, 0x6C, 0x7D, 0x2C, 0x20, 0x7B, 0x22, 0x6E, 0x61, 0x6D, 0x65, 0x22,
    0x3A, 0x20, 0x22, 0x31, 0x22, 0x2C, 0x20, 0x22, 0x66, 0x69, 0x65, 0x6C, 0x64, 0x5F, 0x6E, 0x61,
    0x6D, 0x65, 0x22, 0x3A, 0x20, 0x22, 0x31, 0x22, 0x2C, 0x20, 0x22, 0x70, 0x61, 0x6E, 0x64, 0x61,
    0x73, 0x5F, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x20, 0x22, 0x6C, 0x69, 0x73, 0x74, 0x5B, 0x6C,
    0x69, 0x73, 0x74, 0x5B, 0x75, 0x6E, 0x69, 0x63, 0x6F, 0x64, 0x65, 0x5D, 0x5D, 0x22, 0x2C, 0x20,
    0x22, 0x6E, 0x75, 0x6D, 0x70, 0x79, 0x5F, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x20, 0x22, 0x6F,
    0x62, 0x6A, 0x65, 0x63, 0x74, 0x22, 0x2C, 0x20, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74,
    0x61, 0x22, 0x3A, 0x20, 0x6E, 0x75, 0x6C, 0x6C, 0x7D, 0x2C, 0x20, 0x7B, 0x22, 0x6E, 0x61, 0x6D,
    0x65, 0x22, 0x3A, 0x20, 0x22, 0x32, 0x22, 0x2C, 0x20, 0x22, 0x66, 0x69, 0x65, 0x6C, 0x64, 0x5F,
    0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A, 0x20, 0x22, 0x32, 0x22, 0x2C, 0x20, 0x22, 0x70, 0x61, 0x6E,
    0x64, 0x61, 0x73, 0x5F, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x20, 0x22, 0x6F, 0x62, 0x6A, 0x65,
    0x63, 0x74, 0x22, 0x2C, 0x20, 0x22, 0x6E, 0x75, 0x6D, 0x70, 0x79, 0x5F, 0x74, 0x79, 0x70, 0x65,
    0x22, 0x3A, 0x20, 0x22, 0x6F, 0x62, 0x6A, 0x65, 0x63, 0x74, 0x22, 0x2C, 0x20, 0x22, 0x6D, 0x65,
    0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22, 0x3A, 0x20, 0x6E, 0x75, 0x6C, 0x6C, 0x7D, 0x5D, 0x2C,
    0x20, 0x22, 0x63, 0x72, 0x65, 0x61, 0x74, 0x6F, 0x72, 0x22, 0x3A, 0x20, 0x7B, 0x22, 0x6C, 0x69,
    0x62, 0x72, 0x61, 0x72, 0x79, 0x22, 0x3A, 0x20, 0x22, 0x70, 0x79, 0x61, 0x72, 0x72, 0x6F, 0x77,
    0x22, 0x2C, 0x20, 0x22, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x22, 0x3A, 0x20, 0x22, 0x38,
    0x2E, 0x30, 0x2E, 0x31, 0x22, 0x7D, 0x2C, 0x20, 0x22, 0x70, 0x61, 0x6E, 0x64, 0x61, 0x73, 0x5F,
    0x76, 0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x22, 0x3A, 0x20, 0x22, 0x31, 0x2E, 0x34, 0x2E, 0x33,
    0x22, 0x7D, 0x00, 0x29, 0x5C, 0x1C, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x1C, 0x00,
    0x00, 0x1C, 0x00, 0x00, 0x00, 0x0B, 0x04, 0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  // Read in the data via parquet reader
  cudf::io::parquet_reader_options read_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{reinterpret_cast<const char*>(parquet_data), sizeof(parquet_data)});
  auto result = cudf::io::read_parquet(read_opts);

  // Read in the data via the JSON parser
  auto const cudf_table = cuio_json::detail::parse_nested_json(
    cudf::host_span<SymbolT const>{input.data(), input.size()}, stream_view);

  // Verify that the data read via parquet matches the data read via JSON
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf_table.tbl->view(), result.tbl->view());

  // Verify that the schema read via parquet matches the schema read via JSON
  cudf::test::expect_metadata_equal(cudf_table.metadata, result.metadata);
}

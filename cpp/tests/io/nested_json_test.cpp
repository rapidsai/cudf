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
#include <cudf/io/json.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

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
  constexpr auto stream = cudf::default_stream_value;

  // Test input
  std::string const input = R"(  [{)"
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
  rmm::device_uvector<SymbolT> d_input(input.size(), stream);
  hostdevice_vector<StackSymbolT> stack_context(input.size(), stream);

  ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_input.data(),
                                        input.data(),
                                        input.size() * sizeof(SymbolT),
                                        cudaMemcpyHostToDevice,
                                        stream.value()));

  // Run algorithm
  cuio_json::detail::get_stack_context(d_input, stack_context.device_ptr(), stream);

  // Copy back the results
  stack_context.device_to_host(stream);

  // Make sure we copied back the stack context
  stream.synchronize();

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
  constexpr auto stream = cudf::default_stream_value;

  // Test input
  std::string const input = R"([{"a":{"year":1882,"author": "Bharathi"}, {"a":"filip ʒakotɛ"}}])";

  // Prepare input & output buffers
  rmm::device_uvector<SymbolT> d_input(input.size(), stream);
  hostdevice_vector<StackSymbolT> stack_context(input.size(), stream);

  ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_input.data(),
                                        input.data(),
                                        input.size() * sizeof(SymbolT),
                                        cudaMemcpyHostToDevice,
                                        stream.value()));

  // Run algorithm
  cuio_json::detail::get_stack_context(d_input, stack_context.device_ptr(), stream);

  // Copy back the results
  stack_context.device_to_host(stream);

  // Make sure we copied back the stack context
  stream.synchronize();

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

  // Prepare cuda stream for data transfers & kernels
  constexpr auto stream = cudf::default_stream_value;

  // Default parsing options
  cudf::io::json_reader_options default_options{};

  // Test input
  std::string const input = R"(  [{)"
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
  rmm::device_uvector<SymbolT> d_input(input.size(), stream);

  ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_input.data(),
                                        input.data(),
                                        input.size() * sizeof(SymbolT),
                                        cudaMemcpyHostToDevice,
                                        stream.value()));

  // Parse the JSON and get the token stream
  const auto [d_tokens_gpu, d_token_indices_gpu] =
    cuio_json::detail::get_token_stream(d_input, default_options, stream);

  // Copy back the number of tokens that were written
  thrust::host_vector<PdaTokenT> tokens_gpu =
    cudf::detail::make_host_vector_async(d_tokens_gpu, stream);
  thrust::host_vector<SymbolOffsetT> token_indices_gpu =
    cudf::detail::make_host_vector_async(d_token_indices_gpu, stream);

  // Make sure we copied back all relevant data
  stream.synchronize();

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
  ASSERT_EQ(golden_token_stream.size(), tokens_gpu.size());
  ASSERT_EQ(golden_token_stream.size(), token_indices_gpu.size());

  for (std::size_t i = 0; i < tokens_gpu.size(); i++) {
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
  constexpr auto stream = cudf::default_stream_value;

  // Default parsing options
  cudf::io::json_reader_options default_options{};

  std::string const input = R"( [{"a":0.0, "b":1.0}, {"a":0.1, "b":1.1}, {"a":0.2, "b":1.2}] )";
  // Get the JSON's tree representation
  auto const cudf_table = cuio_json::detail::parse_nested_json(
    cudf::host_span<SymbolT const>{input.data(), input.size()}, default_options, stream);

  auto const expected_col_count  = 2;
  auto const first_column_index  = 0;
  auto const second_column_index = 1;
  EXPECT_EQ(cudf_table.tbl->num_columns(), expected_col_count);

  auto expected_col1 =
    cudf::test::fixed_width_column_wrapper<double>({0.0, 0.1, 0.2}, {true, true, true});
  auto expected_col2 =
    cudf::test::fixed_width_column_wrapper<double>({1.0, 1.1, 1.2}, {true, true, true});
  cudf::column_view parsed_col1 = cudf_table.tbl->get_column(first_column_index);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col1, parsed_col1);
  cudf::column_view parsed_col2 = cudf_table.tbl->get_column(second_column_index);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col2, parsed_col2);
}

TEST_F(JsonTest, UTF_JSON)
{
  // Prepare cuda stream for data transfers & kernels
  constexpr auto stream = cudf::default_stream_value;

  // Default parsing options
  cudf::io::json_reader_options default_options{};

  // Only ASCII string
  std::string const ascii_pass = R"([
  {"a":1,"b":2,"c":[3], "d": {}},
  {"a":1,"b":4.0,"c":[], "d": {"year":1882,"author": "Bharathi"}},
  {"a":1,"b":6.0,"c":[5, 7], "d": null},
  {"a":1,"b":8.0,"c":null, "d": {}},
  {"a":1,"b":null,"c":null},
  {"a":1,"b":Infinity,"c":[null], "d": {"year":-600,"author": "Kaniyan"}}])";

  CUDF_EXPECT_NO_THROW(cuio_json::detail::parse_nested_json(ascii_pass, default_options, stream));

  // utf-8 string that fails parsing.
  std::string const utf_failed = R"([
  {"a":1,"b":2,"c":[3], "d": {}},
  {"a":1,"b":4.0,"c":[], "d": {"year":1882,"author": "Bharathi"}},
  {"a":1,"b":6.0,"c":[5, 7], "d": null},
  {"a":1,"b":8.0,"c":null, "d": {}},
  {"a":1,"b":null,"c":null},
  {"a":1,"b":Infinity,"c":[null], "d": {"year":-600,"author": "filip ʒakotɛ"}}])";
  CUDF_EXPECT_NO_THROW(cuio_json::detail::parse_nested_json(utf_failed, default_options, stream));

  // utf-8 string that passes parsing.
  std::string const utf_pass = R"([
  {"a":1,"b":2,"c":[3], "d": {}},
  {"a":1,"b":4.0,"c":[], "d": {"year":1882,"author": "Bharathi"}},
  {"a":1,"b":6.0,"c":[5, 7], "d": null},
  {"a":1,"b":8.0,"c":null, "d": {}},
  {"a":1,"b":null,"c":null},
  {"a":1,"b":Infinity,"c":[null], "d": {"year":-600,"author": "Kaniyan"}},
  {"a":1,"b":NaN,"c":[null, null], "d": {"year": 2, "author": "filip ʒakotɛ"}}])";
  CUDF_EXPECT_NO_THROW(cuio_json::detail::parse_nested_json(utf_pass, default_options, stream));
}

TEST_F(JsonTest, ExtractColumnWithQuotes)
{
  using cuio_json::SymbolT;

  // Prepare cuda stream for data transfers & kernels
  constexpr auto stream = cudf::default_stream_value;

  // Default parsing options
  cudf::io::json_reader_options options{};
  options.keep_quotes(true);

  std::string const input = R"( [{"a":"0.0", "b":1.0}, {"b":1.1}, {"b":2.1, "a":"2.0"}] )";
  // Get the JSON's tree representation
  auto const cudf_table = cuio_json::detail::parse_nested_json(
    cudf::host_span<SymbolT const>{input.data(), input.size()}, options, stream);

  auto constexpr expected_col_count  = 2;
  auto constexpr first_column_index  = 0;
  auto constexpr second_column_index = 1;
  EXPECT_EQ(cudf_table.tbl->num_columns(), expected_col_count);

  auto expected_col1 =
    cudf::test::strings_column_wrapper({R"("0.0")", R"()", R"("2.0")"}, {true, false, true});
  auto expected_col2 =
    cudf::test::fixed_width_column_wrapper<double>({1.0, 1.1, 2.1}, {true, true, true});
  cudf::column_view parsed_col1 = cudf_table.tbl->get_column(first_column_index);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col1, parsed_col1);
  cudf::column_view parsed_col2 = cudf_table.tbl->get_column(second_column_index);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col2, parsed_col2);
}

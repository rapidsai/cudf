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

#include <io/json/nested_json.h>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <stack>
#include <string>

namespace nested_json = cudf::io::json::gpu;

namespace {

std::string get_node_string(std::size_t const node_id,
                            nested_json::tree_meta_t const& tree_rep,
                            std::string const& json_input)
{
  auto node_to_str = [] __host__ __device__(nested_json::PdaTokenT const token) {
    switch (token) {
      case nested_json::NC_STRUCT: return "STRUCT";
      case nested_json::NC_LIST: return "LIST";
      case nested_json::NC_FN: return "FN";
      case nested_json::NC_STR: return "STR";
      case nested_json::NC_VAL: return "VAL";
      case nested_json::NC_ERR: return "ERR";
      default: return "N/A";
    };
  };

  return "<" + std::to_string(node_id) + ":" + node_to_str(tree_rep.node_categories[node_id]) +
         ":[" + std::to_string(tree_rep.node_range_begin[node_id]) + ", " +
         std::to_string(tree_rep.node_range_end[node_id]) + ") '" +
         json_input.substr(tree_rep.node_range_begin[node_id],
                           tree_rep.node_range_end[node_id] - tree_rep.node_range_begin[node_id]) +
         "'>";
}

void print_tree_representation(std::string const& json_input,
                               nested_json::tree_meta_t const& tree_rep)
{
  for (std::size_t i = 0; i < tree_rep.node_categories.size(); i++) {
    std::size_t parent_id = tree_rep.parent_node_ids[i];
    std::stack<std::size_t> path;
    path.push(i);
    while (parent_id != nested_json::parent_node_sentinel) {
      path.push(parent_id);
      parent_id = tree_rep.parent_node_ids[parent_id];
    }

    while (path.size()) {
      auto const node_id = path.top();
      std::cout << get_node_string(node_id, tree_rep, json_input)
                << (path.size() > 1 ? " -> " : "");
      path.pop();
    }
    std::cout << "\n";
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
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
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
  hostdevice_vector<StackSymbolT> stack_context(input.size(), stream_view);

  ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(
    d_input.data(), input.data(), input.size() * sizeof(SymbolT), cudaMemcpyHostToDevice, stream));

  // Run algorithm
  cudf::io::json::gpu::detail::get_stack_context(d_input, stack_context.device_ptr(), stream);

  // Copy back the results
  stack_context.device_to_host(stream);

  // Make sure we copied back the stack context
  stream_view.synchronize();

  std::vector<char> golden_stack_context{
    '_', '_', '_', '[', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '[', '[', '[', '[', '[', '[', '[', '[', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '[', '[', '[', '[', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '[', '[', '[',
    '{', '[', '[', '[', '[', '[', '[', '[', '{', '{', '{', '{', '{', '[', '{', '{', '[', '[',
    '[', '{', '[', '{', '{', '[', '[', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '[', '_'};

  ASSERT_EQ(golden_stack_context.size(), stack_context.size());
  for (std::size_t i = 0; i < stack_context.size() && i < 1000; i++) {
    ASSERT_EQ(golden_stack_context[i], stack_context[i]);
  }
}

TEST_F(JsonTest, TokenStream)
{
  using cudf::io::json::gpu::PdaTokenT;
  using cudf::io::json::gpu::SymbolOffsetT;
  using cudf::io::json::gpu::SymbolT;

  constexpr std::size_t single_item = 1;

  // Prepare cuda stream for data transfers & kernels
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
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

  ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(
    d_input.data(), input.data(), input.size() * sizeof(SymbolT), cudaMemcpyHostToDevice, stream));

  hostdevice_vector<PdaTokenT> tokens_gpu{input.size(), stream};
  hostdevice_vector<SymbolOffsetT> token_indices_gpu{input.size(), stream};
  hostdevice_vector<SymbolOffsetT> num_tokens_out{single_item, stream};

  // Parse the JSON and get the token stream
  cudf::io::json::gpu::detail::get_token_stream(d_input,
                                                tokens_gpu.device_ptr(),
                                                token_indices_gpu.device_ptr(),
                                                num_tokens_out.device_ptr(),
                                                stream);

  // Copy back the number of tokens that were written
  num_tokens_out.device_to_host(stream);
  tokens_gpu.device_to_host(stream);
  token_indices_gpu.device_to_host(stream);

  // Make sure we copied back all relevant data
  stream_view.synchronize();

  // Golden token stream sample
  using token_t = nested_json::token_t;
  std::vector<std::pair<std::size_t, nested_json::PdaTokenT>> golden_token_stream = {
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
    ASSERT_EQ(golden_token_stream[i].first, token_indices_gpu[i]);
    // Ensure the token category is correct
    ASSERT_EQ(golden_token_stream[i].second, tokens_gpu[i]);
  }
}

TEST_F(JsonTest, TreeRepresentation)
{
  using nested_json::PdaTokenT;
  using nested_json::SymbolOffsetT;
  using nested_json::SymbolT;

  // Prepare cuda stream for data transfers & kernels
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
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

  // Get the JSON's tree representation
  auto tree_rep = nested_json::detail::get_tree_representation(
    cudf::host_span<SymbolT const>{input.data(), input.size()}, stream_view);

  // Print tree representation
  if (std::getenv("CUDA_DBG_DUMP") != nullptr) { print_tree_representation(input, tree_rep); }

  // Golden sample of node categories
  std::vector<nested_json::node_t> golden_node_categories = {
    nested_json::NC_LIST, nested_json::NC_STRUCT, nested_json::NC_FN,     nested_json::NC_STR,
    nested_json::NC_FN,   nested_json::NC_LIST,   nested_json::NC_VAL,    nested_json::NC_VAL,
    nested_json::NC_VAL,  nested_json::NC_FN,     nested_json::NC_STR,    nested_json::NC_FN,
    nested_json::NC_STR,  nested_json::NC_FN,     nested_json::NC_VAL,    nested_json::NC_STRUCT,
    nested_json::NC_FN,   nested_json::NC_STR,    nested_json::NC_FN,     nested_json::NC_LIST,
    nested_json::NC_VAL,  nested_json::NC_STRUCT, nested_json::NC_VAL,    nested_json::NC_STRUCT,
    nested_json::NC_FN,   nested_json::NC_LIST,   nested_json::NC_STRUCT, nested_json::NC_STRUCT,
    nested_json::NC_FN,   nested_json::NC_STR,    nested_json::NC_FN,     nested_json::NC_STR,
    nested_json::NC_FN,   nested_json::NC_VAL};

  // Golden sample of node ids
  std::vector<nested_json::NodeIndexT> golden_parent_node_ids = {nested_json::parent_node_sentinel,
                                                                 0,
                                                                 1,
                                                                 2,
                                                                 1,
                                                                 4,
                                                                 5,
                                                                 5,
                                                                 5,
                                                                 1,
                                                                 9,
                                                                 1,
                                                                 11,
                                                                 1,
                                                                 13,
                                                                 0,
                                                                 15,
                                                                 16,
                                                                 15,
                                                                 18,
                                                                 19,
                                                                 19,
                                                                 19,
                                                                 19,
                                                                 23,
                                                                 24,
                                                                 25,
                                                                 25,
                                                                 15,
                                                                 28,
                                                                 15,
                                                                 30,
                                                                 15,
                                                                 32};

  // Golden sample of node levels
  std::vector<nested_json::TreeDepthT> golden_node_levels = {0, 1, 2, 3, 2, 3, 4, 4, 4, 2, 3, 2,
                                                             3, 2, 3, 1, 2, 3, 2, 3, 4, 4, 4, 4,
                                                             5, 6, 7, 7, 2, 3, 2, 3, 2, 3};

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_begin = {
    2,   3,   5,   17,  29,  38,  39,  41,  44,  49,  59,  72,  81,  108, 116, 124, 126,
    138, 150, 158, 159, 161, 164, 169, 171, 174, 175, 180, 189, 199, 212, 221, 255, 263};

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_end = {
    3,   4,   13,  26,  35,  39,  40,  43,  46,  55,  69,  77,  105, 113, 120, 125, 134,
    147, 155, 159, 160, 162, 168, 170, 172, 175, 176, 181, 195, 209, 217, 252, 260, 267};

  // Check results against golden samples
  ASSERT_EQ(golden_node_categories.size(), tree_rep.node_categories.size());
  ASSERT_EQ(golden_parent_node_ids.size(), tree_rep.parent_node_ids.size());
  ASSERT_EQ(golden_node_levels.size(), tree_rep.node_levels.size());
  ASSERT_EQ(golden_node_range_begin.size(), tree_rep.node_range_begin.size());
  ASSERT_EQ(golden_node_range_end.size(), tree_rep.node_range_end.size());

  for (std::size_t i = 0; i < golden_node_categories.size(); i++) {
    ASSERT_EQ(golden_node_categories[i], tree_rep.node_categories[i]);
    ASSERT_EQ(golden_parent_node_ids[i], tree_rep.parent_node_ids[i]);
    ASSERT_EQ(golden_node_levels[i], tree_rep.node_levels[i]);
    ASSERT_EQ(golden_node_range_begin[i], tree_rep.node_range_begin[i]);
    ASSERT_EQ(golden_node_range_end[i], tree_rep.node_range_end[i]);
  }
}

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

namespace nested_json = cudf::io::json::gpu;

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
  cudf::io::json::gpu::get_stack_context(
    d_input,
    cudf::device_span<StackSymbolT>{stack_context.device_ptr(), stack_context.size()},
    stream);

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
  cudf::io::json::gpu::get_token_stream(
    d_input,
    cudf::device_span<PdaTokenT>{tokens_gpu.device_ptr(), tokens_gpu.size()},
    cudf::device_span<SymbolOffsetT>{token_indices_gpu.device_ptr(), token_indices_gpu.size()},
    num_tokens_out.device_ptr(),
    stream);

  // Copy back the number of tokens that were written
  num_tokens_out.device_to_host(stream);
  tokens_gpu.device_to_host(stream);
  token_indices_gpu.device_to_host(stream);

  // Make sure we copied back all relevant data
  stream_view.synchronize();

  // Golden token stream sample
  std::vector<std::pair<std::size_t, nested_json::PdaTokenT>> golden_token_stream = {
    {2, nested_json::TK_BOL},   {3, nested_json::TK_BOS},   {4, nested_json::TK_BFN},
    {13, nested_json::TK_EFN},  {16, nested_json::TK_BST},  {26, nested_json::TK_EST},
    {28, nested_json::TK_BFN},  {35, nested_json::TK_EFN},  {38, nested_json::TK_BOL},
    {39, nested_json::TK_BOV},  {40, nested_json::TK_POV},  {41, nested_json::TK_BOV},
    {43, nested_json::TK_POV},  {44, nested_json::TK_BOV},  {46, nested_json::TK_POV},
    {46, nested_json::TK_EOL},  {48, nested_json::TK_BFN},  {55, nested_json::TK_EFN},
    {58, nested_json::TK_BST},  {69, nested_json::TK_EST},  {71, nested_json::TK_BFN},
    {77, nested_json::TK_EFN},  {80, nested_json::TK_BST},  {105, nested_json::TK_EST},
    {107, nested_json::TK_BFN}, {113, nested_json::TK_EFN}, {116, nested_json::TK_BOV},
    {120, nested_json::TK_POV}, {120, nested_json::TK_EOS}, {124, nested_json::TK_BOS},
    {125, nested_json::TK_BFN}, {134, nested_json::TK_EFN}, {137, nested_json::TK_BST},
    {147, nested_json::TK_EST}, {149, nested_json::TK_BFN}, {155, nested_json::TK_EFN},
    {158, nested_json::TK_BOL}, {159, nested_json::TK_BOV}, {160, nested_json::TK_POV},
    {161, nested_json::TK_BOS}, {162, nested_json::TK_EOS}, {164, nested_json::TK_BOV},
    {168, nested_json::TK_POV}, {169, nested_json::TK_BOS}, {170, nested_json::TK_BFN},
    {172, nested_json::TK_EFN}, {174, nested_json::TK_BOL}, {175, nested_json::TK_BOS},
    {177, nested_json::TK_EOS}, {180, nested_json::TK_BOS}, {181, nested_json::TK_EOS},
    {182, nested_json::TK_EOL}, {184, nested_json::TK_EOS}, {186, nested_json::TK_EOL},
    {188, nested_json::TK_BFN}, {195, nested_json::TK_EFN}, {198, nested_json::TK_BST},
    {209, nested_json::TK_EST}, {211, nested_json::TK_BFN}, {217, nested_json::TK_EFN},
    {220, nested_json::TK_BST}, {252, nested_json::TK_EST}, {254, nested_json::TK_BFN},
    {260, nested_json::TK_EFN}, {263, nested_json::TK_BOV}, {267, nested_json::TK_POV},
    {267, nested_json::TK_EOS}, {268, nested_json::TK_EOL}};

  // Verify the number of tokens matches
  ASSERT_EQ(golden_token_stream.size(), num_tokens_out[0]);
  
  for (std::size_t i = 0; i < num_tokens_out[0]; i++) {
    // Ensure the index the tokens are pointing to do match
    ASSERT_EQ(golden_token_stream[i].first, token_indices_gpu[i]);
    // Ensure the token category is correct
    ASSERT_EQ(golden_token_stream[i].second, tokens_gpu[i]);
  }
}

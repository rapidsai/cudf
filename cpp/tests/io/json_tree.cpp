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

#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <string>

namespace cuio_json = cudf::io::json;

namespace cudf::io::json {
// Host copy of tree_meta_t
struct tree_meta_t2 {
  std::vector<NodeT> node_categories;
  std::vector<NodeIndexT> parent_node_ids;
  std::vector<TreeDepthT> node_levels;
  std::vector<SymbolOffsetT> node_range_begin;
  std::vector<SymbolOffsetT> node_range_end;
};
}  // namespace cudf::io::json

namespace {
std::string get_node_string(std::size_t const node_id,
                            cuio_json::tree_meta_t2 const& tree_rep,
                            std::string const& json_input)
{
  auto node_to_str = [](cuio_json::PdaTokenT const token) {
    switch (token) {
      case cuio_json::NC_STRUCT: return "STRUCT";
      case cuio_json::NC_LIST: return "LIST";
      case cuio_json::NC_FN: return "FN";
      case cuio_json::NC_STR: return "STR";
      case cuio_json::NC_VAL: return "VAL";
      case cuio_json::NC_ERR: return "ERR";
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
                               cuio_json::tree_meta_t2 const& tree_rep)
{
  for (std::size_t i = 0; i < tree_rep.node_categories.size(); i++) {
    std::size_t parent_id = tree_rep.parent_node_ids[i];
    std::stack<std::size_t> path;
    path.push(i);
    while (parent_id != cuio_json::parent_node_sentinel) {
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

namespace cudf::io::json {
namespace test {

tree_meta_t2 to_cpu_tree(tree_meta_t const& d_value, rmm::cuda_stream_view stream)
{
  return {cudf::detail::make_std_vector_async(d_value.node_categories, stream),
          cudf::detail::make_std_vector_async(d_value.parent_node_ids, stream),
          cudf::detail::make_std_vector_async(d_value.node_levels, stream),
          cudf::detail::make_std_vector_async(d_value.node_range_begin, stream),
          cudf::detail::make_std_vector_async(d_value.node_range_end, stream)};
}

void compare_trees(tree_meta_t2 const& cpu_tree, tree_meta_t const& d_gpu_tree, bool print = false)
{
  auto cpu_num_nodes = cpu_tree.node_categories.size();
  EXPECT_EQ(cpu_num_nodes, d_gpu_tree.node_categories.size());
  EXPECT_EQ(cpu_num_nodes, d_gpu_tree.parent_node_ids.size());
  EXPECT_EQ(cpu_num_nodes, d_gpu_tree.node_levels.size());
  EXPECT_EQ(cpu_num_nodes, d_gpu_tree.node_range_begin.size());
  EXPECT_EQ(cpu_num_nodes, d_gpu_tree.node_range_end.size());
  auto gpu_tree = to_cpu_tree(d_gpu_tree, cudf::default_stream_value);
  // DEBUG prints
  auto to_cat = [](auto v) -> std::string {
    switch (v) {
      case NC_STRUCT: return " S";
      case NC_LIST: return " L";
      case NC_STR: return " \"";
      case NC_VAL: return " V";
      case NC_FN: return " F";
      case NC_ERR: return "ER";
      default: return "UN";
    };
  };
  auto to_int    = [](auto v) { return std::to_string(static_cast<int>(v)); };
  bool mismatch  = false;
  auto print_vec = [&](auto const& cpu, auto const name, auto converter) {
    for (auto const& v : cpu)
      printf("%3s,", converter(v).c_str());
    std::cout << name << std::endl;
  };

#define COMPARE_MEMBER(member)                                                       \
  for (std::size_t i = 0; i < cpu_num_nodes; i++) {                                  \
    EXPECT_EQ(cpu_tree.member[i], gpu_tree.member[i]) << #member << "[" << i << "]"; \
  }
  COMPARE_MEMBER(node_categories);
  COMPARE_MEMBER(parent_node_ids);
  COMPARE_MEMBER(node_levels);
  COMPARE_MEMBER(node_range_begin);
  COMPARE_MEMBER(node_range_end);
#undef COMPARE_MEMBER

#define PRINT_VEC(vec, conv) print_vec(vec, #vec, conv);
#define PRINT_COMPARISON(vec, conv)                                                  \
  PRINT_VEC(cpu_tree.vec, conv);                                                     \
  PRINT_VEC(gpu_tree.vec, conv);                                                     \
  if (!std::equal(cpu_tree.vec.begin(), cpu_tree.vec.end(), gpu_tree.vec.begin())) { \
    for (auto i = 0lu; i < cpu_tree.vec.size(); i++) {                               \
      mismatch |= (gpu_tree.vec[i] != cpu_tree.vec[i]);                              \
      printf("%3s,", (gpu_tree.vec[i] == cpu_tree.vec[i] ? " " : "x"));              \
    }                                                                                \
    printf("\n");                                                                    \
  }
  if (print) {
    for (int i = 0; i < int(cpu_num_nodes); i++)
      printf("%3d,", i);
    printf(" node_id\n");
    PRINT_COMPARISON(node_categories, to_cat);   // Works
    PRINT_COMPARISON(node_levels, to_int);       // Works
    PRINT_COMPARISON(node_range_begin, to_int);  // Works
    PRINT_COMPARISON(node_range_end, to_int);    // Works
    PRINT_COMPARISON(parent_node_ids, to_int);   // Works
    EXPECT_FALSE(mismatch);
  }
#undef PRINT_VEC
#undef PRINT_COMPARISON
}

tree_meta_t2 get_tree_representation_cpu(device_span<PdaTokenT const> tokens_gpu,
                                         device_span<SymbolOffsetT const> token_indices_gpu1,
                                         cudf::io::json_reader_options const& options,
                                         rmm::cuda_stream_view stream)
{
  constexpr bool include_quote_char = true;
  // Copy the JSON tokens to the host
  thrust::host_vector<PdaTokenT> tokens = cudf::detail::make_host_vector_async(tokens_gpu, stream);
  thrust::host_vector<SymbolOffsetT> token_indices =
    cudf::detail::make_host_vector_async(token_indices_gpu1, stream);

  // Make sure tokens have been copied to the host
  stream.synchronize();

  // DEBUG print
  [[maybe_unused]] auto to_token_str = [](PdaTokenT token) {
    switch (token) {
      case token_t::StructBegin: return " {";
      case token_t::StructEnd: return " }";
      case token_t::ListBegin: return " [";
      case token_t::ListEnd: return " ]";
      case token_t::FieldNameBegin: return "FB";
      case token_t::FieldNameEnd: return "FE";
      case token_t::StringBegin: return "SB";
      case token_t::StringEnd: return "SE";
      case token_t::ErrorBegin: return "er";
      case token_t::ValueBegin: return "VB";
      case token_t::ValueEnd: return "VE";
      case token_t::StructMemberBegin: return " <";
      case token_t::StructMemberEnd: return " >";
      default: return ".";
    }
  };
  if (std::getenv("CUDA_DBG_DUMP") != nullptr) {
    std::cout << "Tokens: \n";
    for (auto i = 0u; i < tokens.size(); i++) {
      std::cout << to_token_str(tokens[i]) << " ";
    }
    std::cout << std::endl;
  }

  // Whether a token does represent a node in the tree representation
  auto is_node = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin:
      case token_t::ErrorBegin: return true;
      default: return false;
    };
  };

  // The node that a token represents
  auto token_to_node = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin: return NC_STRUCT;
      case token_t::ListBegin: return NC_LIST;
      case token_t::StringBegin: return NC_STR;
      case token_t::ValueBegin: return NC_VAL;
      case token_t::FieldNameBegin: return NC_FN;
      default: return NC_ERR;
    };
  };

  // Includes quote char for end-of-string token or Skips the quote char for beginning-of-field-name
  auto get_token_index = [include_quote_char](PdaTokenT const token,
                                              SymbolOffsetT const token_index) {
    constexpr SymbolOffsetT quote_char_size = 1;
    switch (token) {
      // Strip off or include quote char for StringBegin
      case token_t::StringBegin: return token_index + (include_quote_char ? 0 : quote_char_size);
      // Strip off or Include trailing quote char for string values for StringEnd
      case token_t::StringEnd: return token_index + (include_quote_char ? quote_char_size : 0);
      // Strip off quote char included for FieldNameBegin
      case token_t::FieldNameBegin: return token_index + quote_char_size;
      default: return token_index;
    };
  };

  // Whether a token expects to be followed by its respective end-of-* token partner
  auto is_begin_of_section = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin: return true;
      default: return false;
    };
  };

  // The end-of-* partner token for a given beginning-of-* token
  auto end_of_partner = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StringBegin: return token_t::StringEnd;
      case token_t::ValueBegin: return token_t::ValueEnd;
      case token_t::FieldNameBegin: return token_t::FieldNameEnd;
      default: return token_t::ErrorBegin;
    };
  };

  // Whether the token pops from the parent node stack
  auto does_pop = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructEnd:
      case token_t::ListEnd: return true;
      default: return false;
    };
  };

  // Whether the token pushes onto the parent node stack
  auto does_push = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin: return true;
      default: return false;
    };
  };

  // The node id sitting on top of the stack becomes the node's parent
  // The full stack represents the path from the root to the current node
  std::stack<std::pair<NodeIndexT, bool>> parent_stack;

  constexpr bool field_name_node    = true;
  constexpr bool no_field_name_node = false;

  std::vector<NodeT> node_categories;
  std::vector<NodeIndexT> parent_node_ids;
  std::vector<TreeDepthT> node_levels;
  std::vector<SymbolOffsetT> node_range_begin;
  std::vector<SymbolOffsetT> node_range_end;

  std::size_t node_id = 0;
  for (std::size_t i = 0; i < tokens.size(); i++) {
    auto token = tokens[i];

    // The section from the original JSON input that this token demarcates
    std::size_t range_begin = get_token_index(token, token_indices[i]);
    std::size_t range_end   = range_begin + 1;

    // Identify this node's parent node id
    std::size_t parent_node_id =
      (parent_stack.size() > 0) ? parent_stack.top().first : parent_node_sentinel;

    // If this token is the beginning-of-{value, string, field name}, also consume the next end-of-*
    // token
    if (is_begin_of_section(token)) {
      if ((i + 1) < tokens.size() && end_of_partner(token) == tokens[i + 1]) {
        // Update the range_end for this pair of tokens
        range_end = get_token_index(tokens[i + 1], token_indices[i + 1]);
        // We can skip the subsequent end-of-* token
        i++;
      }
    }

    // Emit node if this token becomes a node in the tree
    if (is_node(token)) {
      node_categories.push_back(token_to_node(token));
      parent_node_ids.push_back(parent_node_id);
      node_levels.push_back(parent_stack.size());
      node_range_begin.push_back(range_begin);
      node_range_end.push_back(range_end);
    }

    // Modify the stack if needed
    if (token == token_t::FieldNameBegin) {
      parent_stack.push({node_id, field_name_node});
    } else {
      if (does_push(token)) {
        parent_stack.push({node_id, no_field_name_node});
      } else if (does_pop(token)) {
        CUDF_EXPECTS(parent_stack.size() >= 1, "Invalid JSON input.");
        parent_stack.pop();
      }

      // If what we're left with is a field name on top of stack, we need to pop it
      if (parent_stack.size() >= 1 && parent_stack.top().second == field_name_node) {
        parent_stack.pop();
      }
    }

    // Update node_id
    if (is_node(token)) { node_id++; }
  }

  return {std::move(node_categories),
          std::move(parent_node_ids),
          std::move(node_levels),
          std::move(node_range_begin),
          std::move(node_range_end)};
}

}  // namespace test
}  // namespace cudf::io::json

// Base test fixture for tests
struct JsonTest : public cudf::test::BaseFixture {
};

TEST_F(JsonTest, TreeRepresentation)
{
  auto const stream = cudf::default_stream_value;

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
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input = cudf::device_span<cuio_json::SymbolT const>{
    d_scalar.data(), static_cast<size_t>(d_scalar.size())};

  cudf::io::json_reader_options const options{};

  // Parse the JSON and get the token stream
  const auto [tokens_gpu, token_indices_gpu] =
    cudf::io::json::detail::get_token_stream(d_input, options, stream);

  // Get the JSON's tree representation
  auto gpu_tree = cuio_json::detail::get_tree_representation(tokens_gpu, token_indices_gpu, stream);
  // host tree generation
  auto cpu_tree =
    cuio_json::test::get_tree_representation_cpu(tokens_gpu, token_indices_gpu, options, stream);
  // cudf::io::json::test::compare_trees(cpu_tree, gpu_tree);

  // Print tree representation
  if (std::getenv("CUDA_DBG_DUMP") != nullptr) { print_tree_representation(input, cpu_tree); }

  // Golden sample of node categories
  std::vector<cuio_json::node_t> golden_node_categories = {
    cuio_json::NC_LIST, cuio_json::NC_STRUCT, cuio_json::NC_FN,     cuio_json::NC_STR,
    cuio_json::NC_FN,   cuio_json::NC_LIST,   cuio_json::NC_VAL,    cuio_json::NC_VAL,
    cuio_json::NC_VAL,  cuio_json::NC_FN,     cuio_json::NC_STR,    cuio_json::NC_FN,
    cuio_json::NC_STR,  cuio_json::NC_FN,     cuio_json::NC_VAL,    cuio_json::NC_STRUCT,
    cuio_json::NC_FN,   cuio_json::NC_STR,    cuio_json::NC_FN,     cuio_json::NC_LIST,
    cuio_json::NC_VAL,  cuio_json::NC_STRUCT, cuio_json::NC_VAL,    cuio_json::NC_STRUCT,
    cuio_json::NC_FN,   cuio_json::NC_LIST,   cuio_json::NC_STRUCT, cuio_json::NC_STRUCT,
    cuio_json::NC_FN,   cuio_json::NC_STR,    cuio_json::NC_FN,     cuio_json::NC_STR,
    cuio_json::NC_FN,   cuio_json::NC_VAL};

  // Golden sample of node ids
  // clang-format off
  std::vector<cuio_json::NodeIndexT> golden_parent_node_ids = {
    cuio_json::parent_node_sentinel, 0, 1, 2,
    1, 4, 5, 5,
    5, 1, 9, 1,
    11, 1, 13, 0,
    15, 16, 15, 18,
    19, 19, 19, 19,
    23, 24, 25, 25,
    15, 28, 15, 30,
    15, 32};
  // clang-format on

  // Golden sample of node levels
  std::vector<cuio_json::TreeDepthT> golden_node_levels = {0, 1, 2, 3, 2, 3, 4, 4, 4, 2, 3, 2,
                                                           3, 2, 3, 1, 2, 3, 2, 3, 4, 4, 4, 4,
                                                           5, 6, 7, 7, 2, 3, 2, 3, 2, 3};

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_begin = {
    2,   3,   5,   16,  29,  38,  39,  41,  44,  49,  58,  72,  80,  108, 116, 124, 126,
    137, 150, 158, 159, 161, 164, 169, 171, 174, 175, 180, 189, 198, 212, 220, 255, 263};

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_end = {
    3,   4,   13,  27,  35,  39,  40,  43,  46,  55,  70,  77,  106, 113, 120, 125, 134,
    148, 155, 159, 160, 162, 168, 170, 172, 175, 176, 181, 195, 210, 217, 253, 260, 267};

  // Check results against golden samples
  ASSERT_EQ(golden_node_categories.size(), cpu_tree.node_categories.size());
  ASSERT_EQ(golden_parent_node_ids.size(), cpu_tree.parent_node_ids.size());
  ASSERT_EQ(golden_node_levels.size(), cpu_tree.node_levels.size());
  ASSERT_EQ(golden_node_range_begin.size(), cpu_tree.node_range_begin.size());
  ASSERT_EQ(golden_node_range_end.size(), cpu_tree.node_range_end.size());

  for (std::size_t i = 0; i < golden_node_categories.size(); i++) {
    ASSERT_EQ(golden_node_categories[i], cpu_tree.node_categories[i]) << "[" << i << "]";
    ASSERT_EQ(golden_parent_node_ids[i], cpu_tree.parent_node_ids[i]) << "[" << i << "]";
    ASSERT_EQ(golden_node_levels[i], cpu_tree.node_levels[i]) << "[" << i << "]";
    ASSERT_EQ(golden_node_range_begin[i], cpu_tree.node_range_begin[i]) << "[" << i << "]";
    ASSERT_EQ(golden_node_range_end[i], cpu_tree.node_range_end[i]) << "[" << i << "]";
  }
}

TEST_F(JsonTest, TreeRepresentation2)
{
  auto const stream = cudf::default_stream_value;
  // Test input: value end with comma, space, close-brace ", }"
  std::string const input =
    // 0         1         2         3         4         5         6         7         8         9
    // 0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
    R"([ {}, { "a": { "y" : 6, "z": [] }}, { "a" : { "x" : 8, "y": 9 }, "b" : {"x": 10 , "z": 11)"
    "\n}}]";
  // Prepare input & output buffers
  cudf::string_scalar d_scalar(input, true, stream);
  auto d_input = cudf::device_span<cuio_json::SymbolT const>{d_scalar.data(),
                                                             static_cast<size_t>(d_scalar.size())};

  cudf::io::json_reader_options const options{};

  // Parse the JSON and get the token stream
  const auto [tokens_gpu, token_indices_gpu] =
    cudf::io::json::detail::get_token_stream(d_input, options, stream);

  // Get the JSON's tree representation
  auto gpu_tree = cuio_json::detail::get_tree_representation(tokens_gpu, token_indices_gpu, stream);
  // host tree generation
  auto cpu_tree =
    cuio_json::test::get_tree_representation_cpu(tokens_gpu, token_indices_gpu, options, stream);
  cudf::io::json::test::compare_trees(cpu_tree, gpu_tree);

  // Print tree representation
  if (std::getenv("CUDA_DBG_DUMP") != nullptr) { print_tree_representation(input, cpu_tree); }
  // TODO compare with CPU version

  // Golden sample of node categories
  // clang-format off
  std::vector<cuio_json::node_t> golden_node_categories = {
    cuio_json::NC_LIST, cuio_json::NC_STRUCT,
    cuio_json::NC_STRUCT, cuio_json::NC_FN, cuio_json::NC_STRUCT,  cuio_json::NC_FN,  cuio_json::NC_VAL, cuio_json::NC_FN,  cuio_json::NC_LIST,
    cuio_json::NC_STRUCT, cuio_json::NC_FN, cuio_json::NC_STRUCT,  cuio_json::NC_FN,  cuio_json::NC_VAL, cuio_json::NC_FN,  cuio_json::NC_VAL,
                          cuio_json::NC_FN, cuio_json::NC_STRUCT,  cuio_json::NC_FN,  cuio_json::NC_VAL, cuio_json::NC_FN,  cuio_json::NC_VAL};

  // Golden sample of node ids
  std::vector<cuio_json::NodeIndexT> golden_parent_node_ids = {
    cuio_json::parent_node_sentinel, 0,
    0, 2,  3,  4,  5,  4, 7,
    0, 9, 10, 11, 12, 11, 14,
       9, 16, 17, 18, 17, 20};
  // clang-format on

  // Golden sample of node levels
  std::vector<cuio_json::TreeDepthT> golden_node_levels = {
    0, 1, 1, 2, 3, 4, 5, 4, 5, 1, 2, 3, 4, 5, 4, 5, 2, 3, 4, 5, 4, 5,
  };

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_begin = {0,  2,  6,  9,  13, 16, 21, 25, 29, 36, 39,
                                                      44, 47, 52, 56, 60, 66, 71, 73, 77, 83, 87};

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_end = {1,  3,  7,  10, 14, 17, 22, 26, 30, 37, 40,
                                                    45, 48, 53, 57, 61, 67, 72, 74, 79, 84, 89};

  // Check results against golden samples
  ASSERT_EQ(golden_node_categories.size(), cpu_tree.node_categories.size());
  ASSERT_EQ(golden_parent_node_ids.size(), cpu_tree.parent_node_ids.size());
  ASSERT_EQ(golden_node_levels.size(), cpu_tree.node_levels.size());
  ASSERT_EQ(golden_node_range_begin.size(), cpu_tree.node_range_begin.size());
  ASSERT_EQ(golden_node_range_end.size(), cpu_tree.node_range_end.size());

  for (std::size_t i = 0; i < golden_node_categories.size(); i++) {
    ASSERT_EQ(golden_node_categories[i], cpu_tree.node_categories[i]);
    ASSERT_EQ(golden_parent_node_ids[i], cpu_tree.parent_node_ids[i]);
    ASSERT_EQ(golden_node_levels[i], cpu_tree.node_levels[i]);
    ASSERT_EQ(golden_node_range_begin[i], cpu_tree.node_range_begin[i]);
    ASSERT_EQ(golden_node_range_end[i], cpu_tree.node_range_end[i]);
  }
}

TEST_F(JsonTest, TreeRepresentation3)
{
  auto const stream = cudf::default_stream_value;
  // Test input: Json lines with same TreeRepresentation2 input
  std::string const input =
    R"(  {}
 { "a": { "y" : 6, "z": [] }}
 { "a" : { "x" : 8, "y": 9 }, "b" : {"x": 10 , "z": 11 }} )";  // Prepare input & output buffers
  cudf::string_scalar d_scalar(input, true, stream);
  auto d_input = cudf::device_span<cuio_json::SymbolT const>{d_scalar.data(),
                                                             static_cast<size_t>(d_scalar.size())};

  cudf::io::json_reader_options options{};
  options.enable_lines(true);

  // Parse the JSON and get the token stream
  const auto [tokens_gpu, token_indices_gpu] =
    cudf::io::json::detail::get_token_stream(d_input, options, stream);

  // Get the JSON's tree representation
  auto gpu_tree = cuio_json::detail::get_tree_representation(tokens_gpu, token_indices_gpu, stream);
  // host tree generation
  auto cpu_tree =
    cuio_json::test::get_tree_representation_cpu(tokens_gpu, token_indices_gpu, options, stream);
  cudf::io::json::test::compare_trees(cpu_tree, gpu_tree);

  // Print tree representation
  if (std::getenv("CUDA_DBG_DUMP") != nullptr) { print_tree_representation(input, cpu_tree); }
}

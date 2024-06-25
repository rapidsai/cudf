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

#include "io/json/nested_json.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/random.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <thrust/sort.h>

#include <cassert>
#include <string>

namespace cuio_json = cudf::io::json;

struct h_tree_meta_t {
  std::vector<cuio_json::NodeT> node_categories;
  std::vector<cuio_json::NodeIndexT> parent_node_ids;
  std::vector<cuio_json::SymbolOffsetT> node_range_begin;
  std::vector<cuio_json::SymbolOffsetT> node_range_end;
};

struct h_column_tree_csr {
  // position of nnzs
  std::vector<cuio_json::NodeIndexT> rowidx;
  std::vector<cuio_json::NodeIndexT> colidx;
  // node properties
  std::vector<cuio_json::NodeIndexT> column_ids;
  std::vector<cuio_json::NodeT> categories;
  std::vector<cuio_json::SymbolOffsetT> range_begin;
  std::vector<cuio_json::SymbolOffsetT> range_end;
};

bool check_equality(cuio_json::tree_meta_t& d_a,
                    cuio_json::column_tree_csr& d_b,
                    rmm::cuda_stream_view stream)
{
  // convert from tree_meta_t to column_tree_csr
  h_tree_meta_t a{cudf::detail::make_std_vector_async(d_a.node_categories, stream),
                  cudf::detail::make_std_vector_async(d_a.parent_node_ids, stream),
                  cudf::detail::make_std_vector_async(d_a.node_range_begin, stream),
                  cudf::detail::make_std_vector_async(d_a.node_range_end, stream)};

  h_column_tree_csr b{cudf::detail::make_std_vector_async(d_b.rowidx, stream),
                      cudf::detail::make_std_vector_async(d_b.colidx, stream),
                      cudf::detail::make_std_vector_async(d_b.column_ids, stream),
                      cudf::detail::make_std_vector_async(d_b.categories, stream),
                      cudf::detail::make_std_vector_async(d_b.range_begin, stream),
                      cudf::detail::make_std_vector_async(d_b.range_end, stream)};

  stream.synchronize();

  auto num_nodes = a.parent_node_ids.size();
  if (b.rowidx.size() != num_nodes + 1) return false;

  for (auto pos = b.rowidx[0]; pos < b.rowidx[1]; pos++) {
    auto v = b.colidx[pos];
    if (a.parent_node_ids[b.column_ids[v]] != b.column_ids[0]) return false;
  }
  for (size_t u = 1; u < num_nodes; u++) {
    auto v = b.colidx[b.rowidx[u]];
    if (a.parent_node_ids[b.column_ids[u]] != b.column_ids[v]) return false;
    for (auto pos = b.rowidx[u] + 1; pos < b.rowidx[u + 1]; pos++) {
      v = b.colidx[pos];
      if (a.parent_node_ids[b.column_ids[v]] != b.column_ids[u]) return false;
    }
  }
  return true;
}

struct JsonColumnTreeTests : public cudf::test::BaseFixture {};

TEST_F(JsonColumnTreeTests, SimpleLines)
{
  auto const stream = cudf::get_default_stream();
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
  auto const [tokens_gpu, token_indices_gpu] = cudf::io::json::detail::get_token_stream(
    d_input, options, stream, rmm::mr::get_current_device_resource());

  // Get the JSON's tree representation
  auto gpu_tree = cuio_json::detail::get_tree_representation(
    tokens_gpu, token_indices_gpu, false, stream, rmm::mr::get_current_device_resource());

  auto tup =
    cuio_json::detail::records_orient_tree_traversal(d_input,
                                                     gpu_tree,
                                                     false,
                                                     options.is_enabled_lines(),
                                                     stream,
                                                     rmm::mr::get_current_device_resource());
  auto& gpu_col_id      = std::get<0>(tup);
  auto& gpu_row_offsets = std::get<1>(tup);

  auto const num_nodes = gpu_col_id.size();
  rmm::device_uvector<cudf::size_type> sorted_col_ids(gpu_col_id.size(), stream);  // make a copy
  thrust::copy(
    rmm::exec_policy(stream), gpu_col_id.begin(), gpu_col_id.end(), sorted_col_ids.begin());

  // sort by {col_id} on {node_ids} stable
  rmm::device_uvector<cudf::size_type> node_ids(gpu_col_id.size(), stream);
  thrust::sequence(rmm::exec_policy(stream), node_ids.begin(), node_ids.end());
  thrust::stable_sort_by_key(
    rmm::exec_policy(stream), sorted_col_ids.begin(), sorted_col_ids.end(), node_ids.begin());

  cudf::size_type const row_array_parent_col_id = [&]() {
    cudf::size_type value      = cudf::io::json::parent_node_sentinel;
    auto const list_node_index = options.is_enabled_lines() ? 0 : 1;
    CUDF_CUDA_TRY(cudaMemcpyAsync(&value,
                                  gpu_col_id.data() + list_node_index,
                                  sizeof(cudf::size_type),
                                  cudaMemcpyDefault,
                                  stream.value()));
    stream.synchronize();
    return value;
  }();

  auto [d_column_tree, d_unique_col_ids, d_max_row_offsets] =
    cudf::io::json::detail::reduce_to_column_tree(gpu_tree,
                                                  gpu_col_id,
                                                  sorted_col_ids,
                                                  node_ids,
                                                  gpu_row_offsets,
                                                  false,
                                                  row_array_parent_col_id,
                                                  stream);

  auto [d_column_tree_csr, d_max_row_offsets_csr] =
    cudf::io::json::detail::reduce_to_column_tree_csr(gpu_tree,
                                                      gpu_col_id,
                                                      sorted_col_ids,
                                                      node_ids,
                                                      gpu_row_offsets,
                                                      false,
                                                      row_array_parent_col_id,
                                                      stream);

  // assert equality between csr and meta formats
  assert(check_equality(d_column_tree, d_column_tree_csr, stream));
}

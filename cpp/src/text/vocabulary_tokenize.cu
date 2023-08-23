/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <text/subword/bpe_tokenizer.cuh>
#include <text/utilities/tokenize_ops.cuh>

#include <nvtext/tokenize.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuco/static_map.cuh>

namespace nvtext {
namespace detail {
namespace {

template <typename MapRefType>
struct vocabulary_tokenizer_fn {
  cudf::column_device_view const d_strings;
  cudf::string_view d_delimiter;
  MapRefType d_map;
  cudf::size_type default_id;
  cudf::size_type const* d_offsets;
  cudf::size_type* d_results;

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return; }

    auto const d_str = d_strings.element<cudf::string_view>(idx);
    characters_tokenizer tokenizer(d_str, d_delimiter);
    auto d_tokens = d_results + d_offsets[idx];

    cudf::size_type token_idx = 0;
    while (tokenizer.next_token()) {
      auto const pos   = tokenizer.token_byte_positions();
      auto const token = cudf::string_view{d_str.data() + pos.first, (pos.second - pos.first)};
      // lookup token in map
      auto const itr = d_map.find(token);
      auto const id  = (itr != d_map.end()) ? itr->second : default_id;
      // set value into the output
      d_tokens[token_idx++] = id;
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> tokenize_with_vocabulary(cudf::strings_column_view const& input,
                                                       cudf::strings_column_view const& vocabulary,
                                                       cudf::string_scalar const& delimiter,
                                                       cudf::size_type default_id,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");
  CUDF_EXPECTS(not vocabulary.is_empty(), "Parameter vocabulary must not be empty");
  CUDF_EXPECTS(not vocabulary.has_nulls(), "Parameter vocabulary must not have nulls");

  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  // load vocabulary into static-map
  auto d_vocab   = cudf::column_device_view::create(vocabulary.parent(), stream);
  auto vocab_map = std::make_unique<merge_pairs_map_type>(
    static_cast<size_t>(vocabulary.size() * 2),
    cuco::empty_key{-1},
    cuco::empty_value{-1},
    bpe_equal{*d_vocab},
    probe_scheme{bpe_hasher{*d_vocab}},
    hash_table_allocator_type{default_allocator<char>{}, stream},
    stream.value());

  auto iter = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(cudf::size_type idx) { return cuco::make_pair(idx, idx); });
  vocab_map->insert_async(iter, iter + vocabulary.size(), stream.value());

  // count the tokens per string and build the offsets
  auto const d_strings   = cudf::column_device_view::create(input.parent(), stream);
  auto const d_delimiter = delimiter.value(stream);
  auto sizes_itr =
    cudf::detail::make_counting_transform_iterator(0, strings_tokenizer{*d_strings, d_delimiter});
  auto [token_offsets, total_count] =
    cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + input.size(), stream, mr);

  // build output column to hold all token-ids
  auto tokens =
    cudf::make_numeric_column(output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);

  auto map_ref   = vocab_map->ref(cuco::experimental::op::find);
  auto d_offsets = token_offsets->view().data<cudf::size_type>();
  auto d_tokens  = tokens->mutable_view().data<cudf::size_type>();
  vocabulary_tokenizer_fn<decltype(map_ref)> tokenizer{
    *d_strings, d_delimiter, map_ref, default_id, d_offsets, d_tokens};

  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     input.size(),
                     tokenizer);

  return cudf::make_lists_column(input.size(),
                                 std::move(token_offsets),
                                 std::move(tokens),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                 stream,
                                 mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> tokenize_with_vocabulary(cudf::strings_column_view const& input,
                                                       cudf::strings_column_view const& vocabulary,
                                                       cudf::string_scalar const& delimiter,
                                                       cudf::size_type default_id,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::tokenize_with_vocabulary(input, vocabulary, delimiter, default_id, stream, mr);
}

}  // namespace nvtext

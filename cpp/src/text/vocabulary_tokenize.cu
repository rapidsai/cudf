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

#include <text/utilities/tokenize_ops.cuh>

#include <nvtext/tokenize.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/hashing/detail/hash_allocator.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cuco/static_map.cuh>

namespace nvtext {
namespace detail {
namespace {

using string_hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
using hash_value_type    = string_hasher_type::result_type;

/**
 * @brief Hasher function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (string).
 */
struct vocab_hasher {
  cudf::column_device_view const d_strings;
  string_hasher_type hasher{};
  // used by insert
  __device__ hash_value_type operator()(cudf::size_type index) const
  {
    return hasher(d_strings.element<cudf::string_view>(index));
  }
  // used by find
  __device__ hash_value_type operator()(cudf::string_view const& s) const { return hasher(s); }
};

/**
 * @brief Equal function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (string).
 */
struct vocab_equal {
  cudf::column_device_view const d_strings;
  // used by insert
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const noexcept
  {
    return lhs == rhs;  // all rows are expected to be unique
  }
  // used by find
  __device__ bool operator()(cudf::size_type lhs, cudf::string_view const& rhs) const noexcept
  {
    return d_strings.element<cudf::string_view>(lhs) == rhs;
  }
};

using hash_table_allocator_type = rmm::mr::stream_allocator_adaptor<default_allocator<char>>;
using probe_scheme              = cuco::experimental::linear_probing<1, vocab_hasher>;
using vocabulary_map_type       = cuco::experimental::static_map<cudf::size_type,
                                                           cudf::size_type,
                                                           cuco::experimental::extent<std::size_t>,
                                                           cuda::thread_scope_device,
                                                           vocab_equal,
                                                           probe_scheme,
                                                           hash_table_allocator_type>;
}  // namespace
}  // namespace detail

// since column_device_view::create returns is a little more than
// std::unique_ptr<column_device_view> this helper simplifies the return type in a maintainable way
using col_device_view = std::invoke_result_t<decltype(&cudf::column_device_view::create),
                                             cudf::column_view,
                                             rmm::cuda_stream_view>;

struct tokenize_vocabulary::tokenize_vocabulary_impl {
  std::unique_ptr<cudf::column> const vocabulary;
  col_device_view const d_vocabulary;
  std::unique_ptr<detail::vocabulary_map_type> vocabulary_map;

  auto get_map_ref() const { return vocabulary_map->ref(cuco::experimental::op::find); }

  tokenize_vocabulary_impl(std::unique_ptr<cudf::column>&& vocab,
                           col_device_view&& d_vocab,
                           std::unique_ptr<detail::vocabulary_map_type>&& map)
    : vocabulary(std::move(vocab)), d_vocabulary(std::move(d_vocab)), vocabulary_map(std::move(map))
  {
  }
};

struct key_pair {
  __device__ auto operator()(cudf::size_type idx) const noexcept
  {
    return cuco::make_pair(idx, idx);
  }
};

tokenize_vocabulary::tokenize_vocabulary(cudf::strings_column_view const& input,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(not input.is_empty(), "vocabulary must not be empty");
  CUDF_EXPECTS(not input.has_nulls(), "vocabulary must not have nulls");

  // need to hold a copy of the input
  auto vocabulary   = std::make_unique<cudf::column>(input.parent(), stream, mr);
  auto d_vocabulary = cudf::column_device_view::create(vocabulary->view(), stream);

  auto vocab_map = std::make_unique<detail::vocabulary_map_type>(
    static_cast<size_t>(vocabulary->size() * 2),
    cuco::empty_key{-1},
    cuco::empty_value{-1},
    detail::vocab_equal{*d_vocabulary},
    detail::probe_scheme{detail::vocab_hasher{*d_vocabulary}},
    detail::hash_table_allocator_type{default_allocator<char>{}, stream},
    stream.value());

  // the row index is the token id (value for each key in the map)
  auto iter = cudf::detail::make_counting_transform_iterator(0, key_pair{});
  vocab_map->insert_async(iter, iter + vocabulary->size(), stream.value());

  _impl = new tokenize_vocabulary_impl(
    std::move(vocabulary), std::move(d_vocabulary), std::move(vocab_map));
}
tokenize_vocabulary::~tokenize_vocabulary() { delete _impl; }

std::unique_ptr<tokenize_vocabulary> load_vocabulary(cudf::strings_column_view const& input,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return std::make_unique<tokenize_vocabulary>(input, stream, mr);
}

namespace detail {
namespace {

/**
 * @brief Tokenizes each string and uses the map to assign token id values
 *
 * @tparam MapRefType Type of the static_map reference for calling find()
 */
template <typename MapRefType>
struct vocabulary_tokenizer_fn {
  cudf::column_device_view const d_strings;
  cudf::string_view const d_delimiter;
  MapRefType d_map;
  cudf::size_type const default_id;
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
                                                       tokenize_vocabulary const& vocabulary,
                                                       cudf::string_scalar const& delimiter,
                                                       cudf::size_type default_id,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  // count the tokens per string and build the offsets from the counts
  auto const d_strings   = cudf::column_device_view::create(input.parent(), stream);
  auto const d_delimiter = delimiter.value(stream);
  auto const sizes_itr =
    cudf::detail::make_counting_transform_iterator(0, strings_tokenizer{*d_strings, d_delimiter});
  auto [token_offsets, total_count] =
    cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + input.size(), stream, mr);

  // build the output column to hold all the token ids
  auto tokens =
    cudf::make_numeric_column(output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);
  auto map_ref   = vocabulary._impl->get_map_ref();
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
                                                       tokenize_vocabulary const& vocabulary,
                                                       cudf::string_scalar const& delimiter,
                                                       cudf::size_type default_id,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::tokenize_with_vocabulary(input, vocabulary, delimiter, default_id, stream, mr);
}

}  // namespace nvtext

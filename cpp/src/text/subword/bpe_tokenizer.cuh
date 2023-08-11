/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#pragma once

#include <nvtext/bpe_tokenize.hpp>

#include <hash/hash_allocator.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/strings/string_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cuco/static_map.cuh>

#include <cstdint>
#include <type_traits>

namespace nvtext {
namespace detail {

using hash_value_type    = uint32_t;
using string_hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;

/**
 * @brief Hasher function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (string).
 */
struct bpe_hasher {
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
struct bpe_equal {
  cudf::column_device_view const d_strings;
  // used by insert
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const noexcept
  {
    return d_strings.element<cudf::string_view>(lhs) == d_strings.element<cudf::string_view>(rhs);
  }
  // used by find
  __device__ bool operator()(cudf::size_type lhs, cudf::string_view const& rhs) const noexcept
  {
    return d_strings.element<cudf::string_view>(lhs) == rhs;
  }
};

using hash_table_allocator_type = rmm::mr::stream_allocator_adaptor<default_allocator<char>>;

using probe_scheme = cuco::experimental::linear_probing<1, bpe_hasher>;

using merge_pairs_map_type = cuco::experimental::static_map<cudf::size_type,
                                                            cudf::size_type,
                                                            cuco::experimental::extent<std::size_t>,
                                                            cuda::thread_scope_device,
                                                            bpe_equal,
                                                            probe_scheme,
                                                            hash_table_allocator_type>;

}  // namespace detail

// since column_device_view::create returns is a little more than
// std::unique_ptr<column_device_view> this helper simplifies the return type in a more maintainable
// way
using col_device_view = std::invoke_result_t<decltype(&cudf::column_device_view::create),
                                             cudf::column_view,
                                             rmm::cuda_stream_view>;

struct bpe_merge_pairs::bpe_merge_pairs_impl {
  std::unique_ptr<cudf::column> const merge_pairs;
  col_device_view const d_merge_pairs;
  std::unique_ptr<detail::merge_pairs_map_type> merge_pairs_map;

  bpe_merge_pairs_impl(std::unique_ptr<cudf::column>&& merge_pairs,
                       col_device_view&& d_merge_pairs,
                       std::unique_ptr<detail::merge_pairs_map_type>&& merge_pairs_map);

  auto const get_merge_pairs() const { return *d_merge_pairs; }
  auto get_merge_pairs_ref() const { return merge_pairs_map->ref(cuco::experimental::op::find); }
};

}  // namespace nvtext

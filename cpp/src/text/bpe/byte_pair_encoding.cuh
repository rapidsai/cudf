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

#include <nvtext/byte_pair_encoding.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/hashing/detail/hash_allocator.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/strings/string_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cuco/static_map.cuh>

#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/pair.h>

#include <cstdint>
#include <type_traits>

namespace nvtext {
namespace detail {

using string_hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
using hash_value_type    = string_hasher_type::result_type;
using merge_pair_type    = thrust::pair<cudf::string_view, cudf::string_view>;

using hash_table_allocator_type = rmm::mr::stream_allocator_adaptor<default_allocator<char>>;

/**
 * @brief Hasher function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (merge_pair_type).
 *
 * The merge-pairs are in adjacent rows so each index will access two rows of string values.
 * The hash of each string is combined for the returned result.
 */
struct bpe_hasher {
  cudf::column_device_view const d_strings;
  string_hasher_type hasher{};
  // used by insert
  __device__ hash_value_type operator()(cudf::size_type index) const
  {
    index *= 2;
    auto const lhs = d_strings.element<cudf::string_view>(index);
    auto const rhs = d_strings.element<cudf::string_view>(index + 1);
    return cudf::hashing::detail::hash_combine(hasher(lhs), hasher(rhs));
  }
  // used by find
  __device__ hash_value_type operator()(merge_pair_type const& mp) const
  {
    return cudf::hashing::detail::hash_combine(hasher(mp.first), hasher(mp.second));
  }
};

/**
 * @brief Equal function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (merge_pair_type).
 *
 * The merge-pairs are in adjacent rows so each index will access two rows of string values.
 * All rows from the input merge-pairs are unique.
 */
struct bpe_equal {
  cudf::column_device_view const d_strings;
  // used by insert
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const noexcept
  {
    return lhs == rhs;  // all rows are unique
  }
  // used by find
  __device__ bool operator()(cudf::size_type lhs, merge_pair_type const& rhs) const noexcept
  {
    lhs *= 2;
    auto const left  = d_strings.element<cudf::string_view>(lhs);
    auto const right = d_strings.element<cudf::string_view>(lhs + 1);
    return (left == rhs.first) && (right == rhs.second);
  }
};

using bpe_probe_scheme = cuco::experimental::linear_probing<1, bpe_hasher>;

using merge_pairs_map_type = cuco::experimental::static_map<cudf::size_type,
                                                            cudf::size_type,
                                                            cuco::experimental::extent<std::size_t>,
                                                            cuda::thread_scope_device,
                                                            bpe_equal,
                                                            bpe_probe_scheme,
                                                            hash_table_allocator_type>;

/**
 * @brief Hasher function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (merge_pair_type).
 *
 * Each component of the merge-pairs (left and right) are stored individually in the map.
 */
struct mp_hasher {
  cudf::column_device_view const d_strings;
  string_hasher_type hasher{};
  // used by insert
  __device__ hash_value_type operator()(cudf::size_type index) const
  {
    auto const d_str = d_strings.element<cudf::string_view>(index);
    return hasher(d_str);
  }
  // used by find
  __device__ hash_value_type operator()(cudf::string_view const& d_str) const
  {
    return hasher(d_str);
  }
};

/**
 * @brief Equal function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (string).
 */
struct mp_equal {
  cudf::column_device_view const d_strings;
  // used by insert
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const noexcept
  {
    auto const left  = d_strings.element<cudf::string_view>(lhs);
    auto const right = d_strings.element<cudf::string_view>(rhs);
    return left == right;
  }
  // used by find
  __device__ bool operator()(cudf::size_type lhs, cudf::string_view const& rhs) const noexcept
  {
    auto const left = d_strings.element<cudf::string_view>(lhs);
    return left == rhs;
  }
};

using mp_probe_scheme = cuco::experimental::linear_probing<1, mp_hasher>;

using mp_table_map_type = cuco::experimental::static_map<cudf::size_type,
                                                         cudf::size_type,
                                                         cuco::experimental::extent<std::size_t>,
                                                         cuda::thread_scope_device,
                                                         mp_equal,
                                                         mp_probe_scheme,
                                                         hash_table_allocator_type>;

}  // namespace detail

// since column_device_view::create() returns is a little more than
// std::unique_ptr<column_device_view> this helper simplifies the return type for us
using col_device_view = std::invoke_result_t<decltype(&cudf::column_device_view::create),
                                             cudf::column_view,
                                             rmm::cuda_stream_view>;

struct bpe_merge_pairs::bpe_merge_pairs_impl {
  std::unique_ptr<cudf::column> const merge_pairs;
  col_device_view const d_merge_pairs;
  std::unique_ptr<detail::merge_pairs_map_type> merge_pairs_map;  // for BPE
  std::unique_ptr<detail::mp_table_map_type> mp_table_map;        // for locating unpairables

  bpe_merge_pairs_impl(std::unique_ptr<cudf::column>&& merge_pairs,
                       col_device_view&& d_merge_pairs,
                       std::unique_ptr<detail::merge_pairs_map_type>&& merge_pairs_map,
                       std::unique_ptr<detail::mp_table_map_type>&& mp_table_map);

  auto const get_merge_pairs() const { return *d_merge_pairs; }
  auto get_merge_pairs_ref() const { return merge_pairs_map->ref(cuco::experimental::op::find); }
  auto get_mp_table_ref() const { return mp_table_map->ref(cuco::experimental::op::find); }
};

}  // namespace nvtext

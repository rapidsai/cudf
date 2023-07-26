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
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cuco/static_map.cuh>

#include <cstdint>

namespace nvtext {
namespace detail {

using hash_table_allocator_type = rmm::mr::stream_allocator_adaptor<default_allocator<char>>;

using merge_pairs_map_type = cuco::static_map<cudf::hash_value_type,
                                              cudf::size_type,
                                              cuda::thread_scope_device,
                                              hash_table_allocator_type>;

using string_hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;

}  // namespace detail

struct bpe_merge_pairs::bpe_merge_pairs_impl {
  std::unique_ptr<cudf::column> const merge_pairs;
  std::unique_ptr<detail::merge_pairs_map_type> merge_pairs_map;

  bpe_merge_pairs_impl(std::unique_ptr<cudf::column>&& merge_pairs,
                       std::unique_ptr<detail::merge_pairs_map_type>&& merge_pairs_map);

  auto get_merge_pairs() const { return merge_pairs->view(); }
  auto get_merge_pairs_map() const { return merge_pairs_map->get_device_view(); }
};

}  // namespace nvtext

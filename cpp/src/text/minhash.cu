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

#include <nvtext/minhash.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/hashing.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <limits>

namespace nvtext {
namespace detail {
namespace {

struct minhash_fn {
  cudf::column_device_view d_strings;
  cudf::device_span<cudf::hash_value_type const> seeds;
  cudf::size_type width;
  cudf::hash_value_type* d_hashes;

  __device__ void operator()(cudf::size_type idx)
  {
    auto const str_idx  = idx / cudf::detail::warp_size;
    auto const lane_idx = idx % cudf::detail::warp_size;

    if (d_strings.is_null(str_idx)) { return; }

    auto const d_str = d_strings.element<cudf::string_view>(str_idx);

    // initialize hashes output for this string
    if (lane_idx == 0) {
      for (auto seed_idx = 0; seed_idx < static_cast<cudf::size_type>(seeds.size()); ++seed_idx) {
        auto const out_idx = (str_idx * seeds.size()) + seed_idx;
        d_hashes[out_idx]  = d_str.empty() ? 0 : std::numeric_limits<cudf::hash_value_type>::max();
      }
    }
    __syncwarp();

    auto const begin = d_str.begin() + lane_idx;
    auto const end   = [d_str, width = width] {
      auto const length = d_str.length();
      if (length > width) { return (d_str.end() - (width - 1)); }
      return d_str.begin() + static_cast<cudf::size_type>(length > 0);
    }();

    // each lane hashes substrings of parts of the string
    for (auto itr = begin; itr < end; itr += cudf::detail::warp_size) {
      auto const offset = itr.byte_offset();
      auto const ss =
        cudf::string_view(d_str.data() + offset, (itr + width).byte_offset() - offset);

      // hashing each seed on the same section of string is 10x faster than
      // re-substringing (my new word) for each seed
      for (auto seed_idx = 0; seed_idx < static_cast<cudf::size_type>(seeds.size()); ++seed_idx) {
        auto const out_idx = (str_idx * seeds.size()) + seed_idx;
        auto const hasher  = cudf::detail::MurmurHash3_32<cudf::string_view>{seeds[seed_idx]};
        auto const hvalue  = hasher(ss);
        atomicMin(d_hashes + out_idx, hvalue);
      }
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::device_span<cudf::hash_value_type const> seeds,
                                      cudf::size_type width,
                                      cudf::hash_id h_id,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!seeds.empty(), "Parameter seeds cannot be empty", std::invalid_argument);
  CUDF_EXPECTS(
    width > 1, "Parameter width should be an integer value of 2 or greater", std::invalid_argument);
  CUDF_EXPECTS(h_id == cudf::hash_id::HASH_MURMUR3,
               "Only murmur3 hash algorithm supported",
               std::invalid_argument);

  auto output_type = cudf::data_type{cudf::type_to_id<cudf::hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto hashes   = cudf::make_numeric_column(output_type,
                                          input.size() * static_cast<cudf::size_type>(seeds.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  auto d_hashes = hashes->mutable_view().data<cudf::hash_value_type>();

  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<cudf::size_type>(0),
                     input.size() * cudf::detail::warp_size,
                     minhash_fn{*d_strings, seeds, width, d_hashes});

  if (seeds.size() == 1) {
    hashes->set_null_mask(cudf::detail::copy_bitmask(input.parent(), stream, mr),
                          input.null_count());
    return hashes;
  }

  auto offsets = cudf::detail::sequence(
    input.size() + 1,
    cudf::numeric_scalar<cudf::size_type>(0),
    cudf::numeric_scalar<cudf::size_type>(static_cast<cudf::size_type>(seeds.size())),
    stream,
    mr);
  hashes->set_null_mask(rmm::device_buffer{}, 0);  // children have no nulls
  return make_lists_column(input.size(),
                           std::move(offsets),
                           std::move(hashes),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::numeric_scalar<cudf::hash_value_type> seed,
                                      cudf::size_type width,
                                      cudf::hash_id h_id,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto seeds = cudf::device_span<cudf::hash_value_type const>{seed.data(), 1};
  return detail::minhash(input, seeds, width, h_id, cudf::get_default_stream(), mr);
}

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::device_span<cudf::hash_value_type const> seeds,
                                      cudf::size_type width,
                                      cudf::hash_id h_id,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash(input, seeds, width, h_id, cudf::get_default_stream(), mr);
}

}  // namespace nvtext

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
#include <cudf/detail/copy.hpp>
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

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <limits>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Compute the minhash of each string for each seed
 *
 * This is a warp-per-string algorithm where parallel threads within a warp
 * work on substrings of a single string row.
 */
struct minhash_fn {
  cudf::column_device_view d_strings;
  cudf::device_span<cudf::hash_value_type const> seeds;
  cudf::size_type width;
  cudf::hash_value_type* d_hashes;

  __device__ void operator()(std::size_t idx)
  {
    auto const str_idx  = static_cast<cudf::size_type>(idx / cudf::detail::warp_size);
    auto const lane_idx = static_cast<cudf::size_type>(idx % cudf::detail::warp_size);

    if (d_strings.is_null(str_idx)) { return; }

    auto const d_str    = d_strings.element<cudf::string_view>(str_idx);
    auto const d_output = d_hashes + (str_idx * seeds.size());

    // initialize hashes output for this string
    if (lane_idx == 0) {
      auto const init = d_str.empty() ? 0 : std::numeric_limits<cudf::hash_value_type>::max();
      thrust::fill(thrust::seq, d_output, d_output + seeds.size(), init);
    }
    __syncwarp();

    auto const begin = d_str.begin() + lane_idx;
    auto const end   = [d_str, width = width] {
      auto const length = d_str.length();
      if (length > width) { return (d_str.end() - (width - 1)); }
      return d_str.begin() + static_cast<cudf::size_type>(length > 0);
    }();

    // each lane hashes substrings of the given width
    for (auto itr = begin; itr < end; itr += cudf::detail::warp_size) {
      auto const offset = itr.byte_offset();
      auto const hash_str =
        cudf::string_view(d_str.data() + offset, (itr + width).byte_offset() - offset);

      // hashing each seed on the same section of string is 10x faster than
      // computing the substrings for each seed
      for (std::size_t seed_idx = 0; seed_idx < seeds.size(); ++seed_idx) {
        auto const hasher = cudf::detail::MurmurHash3_32<cudf::string_view>{seeds[seed_idx]};
        auto const hvalue = hasher(hash_str);
        atomicMin(d_output + seed_idx, hvalue);
      }
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::device_span<cudf::hash_value_type const> seeds,
                                      cudf::size_type width,
                                      cudf::hash_id hash_function,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!seeds.empty(), "Parameter seeds cannot be empty", std::invalid_argument);
  CUDF_EXPECTS(width >= 2,
               "Parameter width should be an integer value of 2 or greater",
               std::invalid_argument);
  CUDF_EXPECTS(hash_function == cudf::hash_id::HASH_MURMUR3,
               "Only murmur3 hash algorithm supported",
               std::invalid_argument);
  CUDF_EXPECTS(
    (static_cast<std::size_t>(input.size()) * seeds.size()) <
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "The number of seeds times the number of input rows must not exceed maximum of size_type",
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

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::counting_iterator(std::size_t{0}),
    static_cast<std::size_t>(input.size()) * static_cast<std::size_t>(cudf::detail::warp_size),
    minhash_fn{*d_strings, seeds, width, d_hashes});

  if (seeds.size() == 1) {
    hashes->set_null_mask(cudf::detail::copy_bitmask(input.parent(), stream, mr),
                          input.null_count());
    return hashes;
  }

  // build the offsets for the output lists column
  auto offsets = cudf::detail::sequence(
    input.size() + 1,
    cudf::numeric_scalar<cudf::size_type>(0),
    cudf::numeric_scalar<cudf::size_type>(static_cast<cudf::size_type>(seeds.size())),
    stream,
    mr);
  hashes->set_null_mask(rmm::device_buffer{}, 0);  // children have no nulls

  // build the lists column from the offsets and the hashes
  auto result = make_lists_column(input.size(),
                                  std::move(offsets),
                                  std::move(hashes),
                                  input.null_count(),
                                  cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                  stream,
                                  mr);
  // expect this condition to be very rare
  if (input.null_count() > 0) {
    result = cudf::detail::purge_nonempty_nulls(result->view(), stream, mr);
  }
  return result;
}

}  // namespace detail

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::numeric_scalar<cudf::hash_value_type> seed,
                                      cudf::size_type width,
                                      cudf::hash_id hash_function,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto seeds = cudf::device_span<cudf::hash_value_type const>{seed.data(), 1};
  return detail::minhash(input, seeds, width, hash_function, cudf::get_default_stream(), mr);
}

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::device_span<cudf::hash_value_type const> seeds,
                                      cudf::size_type width,
                                      cudf::hash_id hash_function,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash(input, seeds, width, hash_function, cudf::get_default_stream(), mr);
}

}  // namespace nvtext

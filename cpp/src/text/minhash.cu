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
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

struct minhash_fn {
  cudf::column_device_view d_strings;
  cudf::size_type width;
  cudf::hash_value_type seed;

  __device__ cudf::hash_value_type operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) return 0;
    auto const d_str = d_strings.element<cudf::string_view>(idx);

    auto mh = cudf::hash_value_type{0};
    for (cudf::size_type pos = 0; pos < d_str.length() - (width - 1); ++pos) {
      auto const ss     = d_str.substr(pos, width);
      auto const hasher = cudf::detail::MurmurHash3_32<cudf::string_view>{seed};
      auto const hvalue = hasher(ss);
      // cudf::detail::hash_combine(seed, hasher(ss)); matches cudf::hash() result

      mh = mh > 0 ? cudf::detail::min(hvalue, mh) : hvalue;
    }

    return mh;
  }
};

}  // namespace

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::size_type width,
                                      cudf::hash_value_type seed,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(width > 1, "Parameter width should be an integer value of 2 or greater");

  auto output_type = cudf::data_type{cudf::type_to_id<cudf::hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto hashes =
    cudf::make_numeric_column(output_type, input.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  auto d_hashes = hashes->mutable_view().data<cudf::hash_value_type>();

  auto const itr = thrust::make_counting_iterator<cudf::size_type>(0);
  auto const fn  = minhash_fn{*d_strings, width, seed};
  thrust::transform(rmm::exec_policy(stream), itr, itr + input.size(), d_hashes, fn);

  hashes->set_null_mask(cudf::detail::copy_bitmask(input.parent(), stream, mr), input.null_count());

  return hashes;
}

}  // namespace detail

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::size_type width,
                                      cudf::hash_value_type seed,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash(input, width, seed, cudf::get_default_stream(), mr);
}

}  // namespace nvtext

/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/algorithm.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

namespace cudf {
namespace hashing {
namespace detail {

namespace {

using hash_value_type = uint64_t;

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <typename Nullate>
class device_row_hasher {
 public:
  device_row_hasher(Nullate nulls, table_device_view const& t, hash_value_type seed)
    : _check_nulls(nulls), _table(t), _seed(seed)
  {
  }

  __device__ auto operator()(size_type row_index) const noexcept
  {
    return cudf::detail::accumulate(
      _table.begin(),
      _table.end(),
      _seed,
      [row_index, nulls = _check_nulls] __device__(auto hash, auto column) {
        return cudf::type_dispatcher(
          column.type(), element_hasher_adapter{}, column, row_index, nulls, hash);
      });
  }

  /**
   * @brief Computes the hash value of an element in the given column.
   */
  class element_hasher_adapter {
   public:
    template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
    __device__ hash_value_type operator()(column_device_view const& col,
                                          size_type const row_index,
                                          Nullate const _check_nulls,
                                          hash_value_type const _seed) const noexcept
    {
      if (_check_nulls && col.is_null(row_index)) {
        return std::numeric_limits<hash_value_type>::max();
      }
      auto const hasher = XXHash_64<T>{_seed};
      return hasher(col.element<T>(row_index));
    }

    template <typename T, CUDF_ENABLE_IF(not column_device_view::has_element_accessor<T>())>
    __device__ hash_value_type operator()(column_device_view const&,
                                          size_type const,
                                          Nullate const,
                                          hash_value_type const) const noexcept
    {
      CUDF_UNREACHABLE("Unsupported type for XXHash_64");
    }
  };

  Nullate const _check_nulls;
  table_device_view const _table;
  hash_value_type const _seed;
};

}  // namespace

std::unique_ptr<column> xxhash_64(table_view const& input,
                                  uint64_t seed,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(data_type(type_to_id<hash_value_type>()),
                                    input.num_rows(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  bool const nullable   = has_nulls(input);
  auto const input_view = table_device_view::create(input, stream);
  auto output_view      = output->mutable_view();

  // Compute the hash value for each row
  thrust::tabulate(rmm::exec_policy(stream),
                   output_view.begin<hash_value_type>(),
                   output_view.end<hash_value_type>(),
                   device_row_hasher(nullable, *input_view, seed));

  return output;
}

}  // namespace detail

std::unique_ptr<column> xxhash_64(table_view const& input,
                                  uint64_t seed,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::xxhash_64(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf

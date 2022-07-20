/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/hashing.hpp>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
class device_spark_row_hasher {
  friend class cudf::experimental::row::hash::row_hasher<
    device_spark_row_hasher>;  ///< Allow row_hasher to access private members.

 public:
  device_spark_row_hasher() = delete;

  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ auto operator()(size_type row_index) const noexcept
  {
    return detail::accumulate(
      _table.begin(),
      _table.end(),
      _seed,
      [row_index, nulls = this->_check_nulls] __device__(auto hash, auto column) {
        return cudf::type_dispatcher(column.type(),
                                     element_hasher_adapter<hash_function>{nulls, hash, hash},
                                     column,
                                     row_index);
      });
  }

 private:
  /**
   * @brief Computes the hash value of an element in the given column.
   *
   * When the column is non-nested, this is a simple wrapper around the element_hasher.
   * When the column is nested, this uses the element_hasher to hash the shape and values of the
   * column.
   */
  template <template <typename> class hash_fn>
  class element_hasher_adapter {
   public:
    __device__ element_hasher_adapter(Nullate check_nulls,
                                      uint32_t seed,
                                      hash_value_type null_hash) noexcept
      : _check_nulls(check_nulls), _seed(seed), _null_hash(null_hash)
    {
    }

    template <typename T, CUDF_ENABLE_IF(not cudf::is_nested<T>())>
    __device__ hash_value_type operator()(column_device_view const& col,
                                          size_type row_index) const noexcept
    {
      auto const hasher = cudf::experimental::row::hash::element_hasher<hash_fn, Nullate>(
        _check_nulls, _seed, _null_hash);
      return hasher.template operator()<T>(col, row_index);
    }

    template <typename T, CUDF_ENABLE_IF(cudf::is_nested<T>())>
    __device__ hash_value_type operator()(column_device_view const& col,
                                          size_type row_index) const noexcept
    {
      column_device_view curr_col = col.slice(row_index, 1);
      while (is_nested(curr_col.type())) {
        if (curr_col.type().id() == type_id::STRUCT) {
          if (curr_col.num_child_columns() == 0) { return _seed; }
          // Non-empty structs are assumed to be decomposed and contain only one child
          curr_col = detail::structs_column_device_view(curr_col).get_sliced_child(0);
        } else if (curr_col.type().id() == type_id::LIST) {
          curr_col = detail::lists_column_device_view(curr_col).get_sliced_child();
        }
      }

      return detail::accumulate(
        thrust::counting_iterator(0),
        thrust::counting_iterator(curr_col.size()),
        _seed,
        [curr_col, nulls = this->_check_nulls] __device__(auto hash, auto element_index) {
          auto const hasher =
            cudf::experimental::row::hash::element_hasher<hash_fn, Nullate>(nulls, hash, hash);
          return cudf::type_dispatcher<cudf::experimental::dispatch_void_if_nested>(
            curr_col.type(), hasher, curr_col, element_index);
        });
    }

    Nullate const _check_nulls;        ///< Whether to check for nulls
    uint32_t const _seed;              ///< The seed to use for hashing
    hash_value_type const _null_hash;  ///< Hash value to use for null elements
  };

  CUDF_HOST_DEVICE device_spark_row_hasher(Nullate check_nulls,
                                           table_device_view t,
                                           uint32_t seed = DEFAULT_HASH_SEED) noexcept
    : _check_nulls{check_nulls}, _table{t}, _seed(seed)
  {
  }

  Nullate const _check_nulls;
  table_device_view const _table;
  uint32_t const _seed;
};

}  // namespace

std::unique_ptr<column> spark_murmur_hash3_32(table_view const& input,
                                              uint32_t seed,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  // TODO: Spark uses int32_t hash values, but libcudf defines hash_value_type
  // as uint32_t elsewhere. This should be investigated and unified. I suspect
  // we should use int32_t everywhere. Also check this for hash seeds. --bdice
  using hash_value_type = int32_t;

  auto output = make_numeric_column(data_type(type_to_id<hash_value_type>()),
                                    input.num_rows(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  bool const nullable = has_nulls(input);
  auto const row_hasher =
    cudf::experimental::row::hash::row_hasher<device_spark_row_hasher>(input, stream);
  auto output_view = output->mutable_view();

  // Compute the hash value for each row
  thrust::tabulate(rmm::exec_policy(stream),
                   output_view.begin<hash_value_type>(),
                   output_view.end<hash_value_type>(),
                   row_hasher.device_hasher<SparkMurmurHash3_32>(nullable, seed));

  return output;
}

}  // namespace detail
}  // namespace cudf

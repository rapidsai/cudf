/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <sort/sort_impl.cuh>

#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

using namespace THRUST_WRAPPED_NAMESPACE;
namespace cudf {
namespace detail {
namespace {

/**
 * @brief Type-dispatched functor for sorting a single column.
 */
struct column_sorted_order_fn {
  /**
   * @brief Compile time check for allowing radix sort for column type.
   *
   * Floating point is removed here for special handling of NaNs.
   */
  template <typename T>
  static constexpr bool is_radix_sort_supported()
  {
    return cudf::is_fixed_width<T>() && !cudf::is_floating_point<T>();
  }

  /**
   * @brief Sorts fixed-width columns using faster thrust sort.
   *
   * @param input Column to sort
   * @param indices Output sorted indices
   * @param ascending True if sort order is ascending
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  template <typename T, std::enable_if_t<is_radix_sort_supported<T>()>* = nullptr>
  void radix_sort(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  rmm::cuda_stream_view stream)
  {
    // A non-stable sort on a column of arithmetic type with no nulls will use a radix sort
    // if specifying only the `thrust::less` or `thrust::greater` comparators.
    // But this also requires making a copy of the input data.
    auto temp_col = column(input, stream);
    auto d_col    = temp_col.mutable_view();
    if (ascending) {
      thrust::sort_by_key(rmm::exec_policy(stream),
                          d_col.begin<T>(),
                          d_col.end<T>(),
                          indices.begin<size_type>(),
                          thrust::less<T>());
    } else {
      thrust::sort_by_key(rmm::exec_policy(stream),
                          d_col.begin<T>(),
                          d_col.end<T>(),
                          indices.begin<size_type>(),
                          thrust::greater<T>());
    }
  }
  template <typename T, std::enable_if_t<!is_radix_sort_supported<T>()>* = nullptr>
  void radix_sort(column_view const&, mutable_column_view&, bool, rmm::cuda_stream_view)
  {
    CUDF_FAIL("Only fixed-width types are suitable for faster sorting");
  }

  /**
   * @brief Sorts a single column with a relationally comparable type.
   *
   * This includes numeric, timestamp, duration, and string types.
   *
   * @param input Column to sort
   * @param indices Output sorted indices
   * @param ascending True if sort order is ascending
   * @param null_precedence How null rows are to be ordered
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  template <typename T, std::enable_if_t<cudf::is_relationally_comparable<T, T>()>* = nullptr>
  void operator()(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  null_order null_precedence,
                  rmm::cuda_stream_view stream)
  {
    // column with nulls or non-supported types will also use a comparator
    if (input.has_nulls() || !is_radix_sort_supported<T>()) {
      auto keys = column_device_view::create(input, stream);
      thrust::sort(rmm::exec_policy(stream),
                   indices.begin<size_type>(),
                   indices.end<size_type>(),
                   simple_comparator<T>{*keys, input.has_nulls(), ascending, null_precedence});
    } else {
      radix_sort<T>(input, indices, ascending, stream);
    }
  }

  template <typename T, std::enable_if_t<!cudf::is_relationally_comparable<T, T>()>* = nullptr>
  void operator()(column_view const&, mutable_column_view&, bool, null_order, rmm::cuda_stream_view)
  {
    CUDF_FAIL("Column type must be relationally comparable");
  }
};

}  // namespace

/**
 * @copydoc
 * sorted_order(column_view&,order,null_order,rmm::cuda_stream_view,rmm::mr::device_memory_resource*)
 */
template <>
std::unique_ptr<column> sorted_order<false>(column_view const& input,
                                            order column_order,
                                            null_order null_precedence,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  auto sorted_indices = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
  mutable_column_view indices_view = sorted_indices->mutable_view();
  thrust::sequence(
    rmm::exec_policy(stream), indices_view.begin<size_type>(), indices_view.end<size_type>(), 0);
  cudf::type_dispatcher<dispatch_storage_type>(input.type(),
                                               column_sorted_order_fn{},
                                               input,
                                               indices_view,
                                               column_order == order::ASCENDING,
                                               null_precedence,
                                               stream);
  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf

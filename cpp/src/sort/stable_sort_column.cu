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

#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cudf {
namespace detail {
namespace {

struct column_stable_sorted_order_fn {
  /**
   * @brief Stable sort of fixed-width columns using a thrust sort with no comparator.
   *
   * @param input Column to sort
   * @param indices Output sorted indices
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  template <typename T, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  void faster_stable_sort(column_view const& input,
                          mutable_column_view& indices,
                          rmm::cuda_stream_view stream)
  {
    auto temp_col = column(input, stream);
    auto d_col    = temp_col.mutable_view();
    thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream),
                               d_col.begin<T>(),
                               d_col.end<T>(),
                               indices.begin<size_type>());
  }
  template <typename T, std::enable_if_t<!cudf::is_fixed_width<T>()>* = nullptr>
  void faster_stable_sort(column_view const&, mutable_column_view&, rmm::cuda_stream_view)
  {
    CUDF_FAIL("Only fixed-width types are suitable for faster stable sorting");
  }

  /**
   * @brief Stable sorts a single column with a relationally comparable type.
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
    if (!ascending || input.has_nulls() || !cudf::is_fixed_width<T>()) {
      auto keys = column_device_view::create(input, stream);
      thrust::stable_sort(
        rmm::exec_policy_nosync(stream),
        indices.begin<size_type>(),
        indices.end<size_type>(),
        simple_comparator<T>{*keys, input.has_nulls(), ascending, null_precedence});
    } else {
      faster_stable_sort<T>(input, indices, stream);
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
std::unique_ptr<column> sorted_order<true>(column_view const& input,
                                           order column_order,
                                           null_order null_precedence,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  auto sorted_indices = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
  mutable_column_view indices_view = sorted_indices->mutable_view();
  thrust::sequence(rmm::exec_policy_nosync(stream),
                   indices_view.begin<size_type>(),
                   indices_view.end<size_type>(),
                   0);
  cudf::type_dispatcher<dispatch_storage_type>(input.type(),
                                               column_stable_sorted_order_fn{},
                                               input,
                                               indices_view,
                                               column_order == order::ASCENDING,
                                               null_precedence,
                                               stream);
  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf

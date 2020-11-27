/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <rmm/cuda_stream_view.hpp>
#include <vector>

namespace cudf {
namespace detail {
namespace {

template <typename InputIterator, typename BinaryPredicate>
struct unique_copy_fn {
  /**
   * @brief Functor for unique_copy()
   *
   * The logic here is equivalent to:
   * @code
   *   ((keep == duplicate_keep_option::KEEP_LAST) ||
   *    (i == 0 || !comp(iter[i], iter[i - 1]))) &&
   *   ((keep == duplicate_keep_option::KEEP_FIRST) ||
   *    (i == last_index || !comp(iter[i], iter[i + 1])))
   * @endcode
   *
   * It is written this way so that the `comp` comparator
   * function appears only once minimizing the inlining
   * required and reducing the compile time.
   */
  __device__ bool operator()(size_type i)
  {
    size_type boundary = 0;
    size_type offset   = 1;
    auto keep_option   = duplicate_keep_option::KEEP_LAST;
    do {
      if ((keep != keep_option) && (i != boundary) && comp(iter[i], iter[i - offset])) {
        return false;
      }
      keep_option = duplicate_keep_option::KEEP_FIRST;
      boundary    = last_index;
      offset      = -offset;
    } while (offset < 0);
    return true;
  }

  InputIterator iter;
  duplicate_keep_option const keep;
  BinaryPredicate comp;
  size_type const last_index;
};

}  // namespace

/**
 * @brief Copies unique elements from the range [first, last) to output iterator `output`.
 *
 * In a consecutive group of duplicate elements, depending on parameter `keep`,
 * only the first element is copied, or the last element is copied or neither is copied.
 *
 * @return End of the range to which the elements are copied.
 */
template <typename InputIterator, typename OutputIterator, typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate comp,
                           duplicate_keep_option const keep,
                           rmm::cuda_stream_view stream)
{
  size_type const last_index = thrust::distance(first, last) - 1;
  return thrust::copy_if(
    rmm::exec_policy(stream)->on(stream.value()),
    first,
    last,
    thrust::counting_iterator<size_type>(0),
    output,
    unique_copy_fn<InputIterator, BinaryPredicate>{first, keep, comp, last_index});
}

/**
 * @brief Create a column_view of index values which represent the row values
 * without duplicates as per @p `keep`
 *
 * Given a `keys` table_view, each row index is copied to output `unique_indices`, if the
 * corresponding row of `keys` table_view is unique, where the definition of unique depends on the
 * value of @p keep:
 * - KEEP_FIRST: only the first of a sequence of duplicate rows is copied
 * - KEEP_LAST: only the last of a sequence of duplicate rows is copied
 * - KEEP_NONE: only unique rows are kept
 *
 * @param[in] keys            table_view to identify duplicate rows
 * @param[out] unique_indices Column to store the index with unique rows
 * @param[in] keep            keep first entry, last entry, or no entries if duplicates found
 * @param[in] nulls_equal     flag to denote nulls are equal if null_equality::EQUAL,
 *                            nulls are not equal if null_equality::UNEQUAL
 * @param[in] stream          CUDA stream used for device memory operations and kernel launches.
 *
 * @return column_view column_view of unique row index as per specified `keep`, this is actually
 * slice of `unique_indices`.
 */
column_view get_unique_ordered_indices(cudf::table_view const& keys,
                                       cudf::mutable_column_view& unique_indices,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       rmm::cuda_stream_view stream)
{
  // sort only indices
  auto sorted_indices = sorted_order(keys,
                                     std::vector<order>{},
                                     std::vector<null_order>{},
                                     stream,
                                     rmm::mr::get_current_device_resource());

  // extract unique indices
  auto device_input_table = cudf::table_device_view::create(keys, stream.value());

  if (cudf::has_nulls(keys)) {
    auto comp = row_equality_comparator<true>(
      *device_input_table, *device_input_table, nulls_equal == null_equality::EQUAL);
    auto result_end = unique_copy(sorted_indices->view().begin<cudf::size_type>(),
                                  sorted_indices->view().end<cudf::size_type>(),
                                  unique_indices.begin<cudf::size_type>(),
                                  comp,
                                  keep,
                                  stream);

    return cudf::detail::slice(
      column_view(unique_indices),
      0,
      thrust::distance(unique_indices.begin<cudf::size_type>(), result_end));
  } else {
    auto comp = row_equality_comparator<false>(
      *device_input_table, *device_input_table, nulls_equal == null_equality::EQUAL);
    auto result_end = unique_copy(sorted_indices->view().begin<cudf::size_type>(),
                                  sorted_indices->view().end<cudf::size_type>(),
                                  unique_indices.begin<cudf::size_type>(),
                                  comp,
                                  keep,
                                  stream);

    return cudf::detail::slice(
      column_view(unique_indices),
      0,
      thrust::distance(unique_indices.begin<cudf::size_type>(), result_end));
  }
}

std::unique_ptr<table> drop_duplicates(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  if (0 == input.num_rows() || 0 == input.num_columns() || 0 == keys.size()) {
    return empty_like(input);
  }

  auto keys_view = input.select(keys);

  // The values will be filled into this column
  auto unique_indices = cudf::make_numeric_column(
    data_type{type_id::INT32}, keys_view.num_rows(), mask_state::UNALLOCATED, stream);
  auto mutable_unique_indices_view = unique_indices->mutable_view();
  // This is just slice of `unique_indices` but with different size as per the
  // keys_view has been processed in `get_unique_ordered_indices`
  auto unique_indices_view = detail::get_unique_ordered_indices(
    keys_view, mutable_unique_indices_view, keep, nulls_equal, stream);

  // run gather operation to establish new order
  return detail::gather(input,
                        unique_indices_view,
                        detail::out_of_bounds_policy::NULLIFY,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

}  // namespace detail

std::unique_ptr<table> drop_duplicates(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option const keep,
                                       null_equality nulls_equal,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::drop_duplicates(input, keys, keep, nulls_equal, rmm::cuda_stream_default, mr);
}

}  // namespace cudf

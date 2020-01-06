/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/gather.hpp>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <cmath>

namespace cudf {
namespace experimental {
namespace detail {

/*
 * unique_copy copies elements from the range [first, last) to a range beginning
 * with output, except that in a consecutive group of duplicate elements only
 * depending on last argument keep, only the first one is copied, or the last
 * one is copied or neither is copied. The return value is the end of the range
 * to which the elements are copied.
 */
template<typename Exec,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
  OutputIterator unique_copy(Exec&& exec,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output,
                             BinaryPredicate comp,
                             const duplicate_keep_option keep)
{
  size_type last_index = thrust::distance(first,last)-1;
  if (keep == duplicate_keep_option::KEEP_NONE) {
    return thrust::copy_if(exec,
               first,
               last,
               thrust::counting_iterator<size_type>(0),
               output,
               [first, comp, last_index] __device__ (size_type i) {
               return (i == 0 || !comp(first[i], first[i-1]))
                   && (i == last_index || !comp(first[i], first[i+1]));
               });
  } else {
    size_type offset = 1;
    if (keep == duplicate_keep_option::KEEP_FIRST) {
      last_index = 0;
      offset = -1;
    }
    return thrust::copy_if(exec,
               first,
               last,
               thrust::counting_iterator<size_type>(0),
               output,
               [first, comp, last_index, offset] __device__ (size_type i) {
                 return (i == last_index || !comp(first[i], first[i+offset]));
               });
   }
}

/**
 * @brief Create a column_view of index values which represent the row values
 * without duplicates as per @p `keep`
 *
 * Given a `keys` table_view, each row index is copied to output `unique_indices`, if the corresponding
 * row of `keys` table_view is unique, where the definition of unique depends on the value of @p keep:
 * - KEEP_FIRST: only the first of a sequence of duplicate rows is copied
 * - KEEP_LAST: only the last of a sequence of duplicate rows is copied
 * - KEEP_NONE: only unique rows are kept
 *
 * @param[in] keys            table_view to identify duplicate rows
 * @param[out] unique_indices Column to store the index with unique rows
 * @param[in] keep            keep first entry, last entry, or no entries if duplicates found
 * @param[in] nulls_are_equal flag to denote nulls are equal if true,
 * nulls are not equal if false
 * @param[in] mr Optional, The resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to execute kernels
 *
 * @return column_view column_view of unique row index as per specified `keep`, this is actually slice of `unique_indices`.
 */
column_view get_unique_ordered_indices(cudf::table_view const& keys,
                                       cudf::mutable_column_view & unique_indices,
                                       duplicate_keep_option const& keep,
                                       bool const& nulls_are_equal = true,
                                       cudaStream_t stream=0)
{
  // sort only indices
  auto sorted_indices = sorted_order(keys,
                                     std::vector<order>{},
                                     std::vector<null_order>{},
                                     rmm::mr::get_default_resource(),
                                     stream);


  // extract unique indices 
  auto device_input_table = cudf::table_device_view::create(keys, stream);

  if(cudf::has_nulls(keys)) {
    auto comp = row_equality_comparator<true>(*device_input_table,
                                              *device_input_table,
                                               nulls_are_equal);
    auto result_end = unique_copy(rmm::exec_policy(stream)->on(stream),
                                  sorted_indices->view().begin<cudf::size_type>(),
                                  sorted_indices->view().end<cudf::size_type>(),
                                  unique_indices.begin<cudf::size_type>(),
                                  comp,
                                  keep);
  
    return cudf::experimental::detail::slice(column_view(unique_indices), 0,
            thrust::distance(unique_indices.begin<cudf::size_type>(), result_end));
  } else {
    auto comp = row_equality_comparator<false>(*device_input_table,
                                               *device_input_table,
                                               nulls_are_equal);
    auto result_end = unique_copy(rmm::exec_policy(stream)->on(stream),
                                  sorted_indices->view().begin<cudf::size_type>(),
                                  sorted_indices->view().end<cudf::size_type>(),
                                  unique_indices.begin<cudf::size_type>(),
                                  comp,
                                  keep);
  
    return cudf::experimental::detail::slice(column_view(unique_indices), 0,
            thrust::distance(unique_indices.begin<cudf::size_type>(), result_end));
  }
  
}

cudf::size_type unique_count(table_view const& keys,
                             bool const& nulls_are_equal = true,
                             cudaStream_t stream=0)
{
  // sort only indices
  auto sorted_indices = sorted_order(keys,
                                     std::vector<order>{},
                                     std::vector<null_order>{},
                                     rmm::mr::get_default_resource(),
                                     stream);
  
  // count unique elements
  auto sorted_row_index = sorted_indices->view().data<cudf::size_type>();
  auto device_input_table = cudf::table_device_view::create(keys, stream);

  if(cudf::has_nulls(keys)) {
    row_equality_comparator<true> comp (*device_input_table,
                                              *device_input_table,
                                              nulls_are_equal);
    return thrust::count_if(rmm::exec_policy(stream)->on(stream),
              thrust::counting_iterator<cudf::size_type>(0),
              thrust::counting_iterator<cudf::size_type>(keys.num_rows()),
              [sorted_row_index, comp]
              __device__ (cudf::size_type i) {
              return (i == 0 || not comp(sorted_row_index[i], sorted_row_index[i-1]));
              });
  } else {
    row_equality_comparator<false> comp(*device_input_table,
                                              *device_input_table,
                                              nulls_are_equal);
    return thrust::count_if(rmm::exec_policy(stream)->on(stream),
              thrust::counting_iterator<cudf::size_type>(0),
              thrust::counting_iterator<cudf::size_type>(keys.num_rows()),
              [sorted_row_index, comp]
              __device__ (cudf::size_type i) {
              return (i == 0 || not comp(sorted_row_index[i], sorted_row_index[i-1]));
              });
  }
}

std::unique_ptr<experimental::table>
  drop_duplicates(table_view const& input,
                  std::vector<size_type> const& keys,
                  duplicate_keep_option const& keep,
                  bool const& nulls_are_equal,
                  rmm::mr::device_memory_resource* mr,
                  cudaStream_t stream)
{
  if (0 == input.num_rows() || 
      0 == input.num_columns() ||
      0 == keys.size()
      ) {
      return experimental::empty_like(input);
  }

  auto keys_view = input.select(keys);
  
  // The values will be filled into this column
  auto unique_indices = 
        cudf::make_numeric_column(data_type{INT32}, 
                                  keys_view.num_rows(), UNALLOCATED, stream, mr);
  auto mutable_unique_indices_view = unique_indices->mutable_view();
  // This is just slice of `unique_indices` but with different size as per the
  // keys_view has been processed in `get_unique_ordered_indices`
  auto unique_indices_view = 
      detail::get_unique_ordered_indices(keys_view,
                                         mutable_unique_indices_view,
                                         keep, nulls_are_equal,
                                         stream);
 
  // run gather operation to establish new order
  return detail::gather(input, unique_indices_view, false, false, false, mr, stream);
}

cudf::size_type unique_count(column_view const& input,
                             bool const& ignore_nulls,
                             bool const& nan_as_null,
                             cudaStream_t stream)
{
  if (0 == input.size() || input.null_count() == input.size()) {
    return 0;
  }

  cudf::size_type nrows = input.size();
 
  bool has_nan = false;
  // Check for Nans
  // Checking for nulls in input and flag nan_as_null, as the count will
  // only get affected if these two conditions are true. NAN will only be
  // be an extra if nan_as_null was true and input also had null, which
  // will increase the count by 1.
  if(input.has_nulls() and nan_as_null){
      has_nan = cudf::experimental::type_dispatcher(input.type(), has_nans{}, input, stream);
  }

  auto count = detail::unique_count(table_view{{input}}, true, stream);

  // if nan is considered null and there are already null values
  if (nan_as_null and has_nan and input.has_nulls())
    --count;

  if(ignore_nulls and input.has_nulls())
    return --count;
  else
    return count;
}

}// namespace detail

std::unique_ptr<experimental::table>
  drop_duplicates(table_view const& input,
                  std::vector<size_type> const& keys,
                  duplicate_keep_option const& keep,
                  bool const& nulls_are_equal,
                  rmm::mr::device_memory_resource* mr) {

    return detail::drop_duplicates(input, keys, keep, nulls_are_equal, mr);
}

cudf::size_type unique_count(column_view const& input,
                             bool const& ignore_nulls,
                             bool const& nan_as_null,
                             rmm::mr::device_memory_resource *mr) {

    return detail::unique_count(input, ignore_nulls, nan_as_null);
}

}// namespace experimental
}// namespace cudf

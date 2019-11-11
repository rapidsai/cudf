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
#include <cudf/detail/copy_if.cuh>
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
         typename BinaryPredicate,
    typename IndexType = typename
  thrust::iterator_difference<InputIterator>::type>
  OutputIterator unique_copy(Exec&& exec,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output,
                             BinaryPredicate comp,
                             const duplicate_keep_option keep)
{
  IndexType n = (last-first)-1;
  if (keep == duplicate_keep_option::KEEP_FIRST) {
      return thrust::copy_if(exec,
              first,
              last,
              thrust::counting_iterator<IndexType>(0),
              output, 
              [first, comp, n] __device__ (IndexType i) mutable {
              return (i == 0 || !comp(first[i], first[i-1]));
              }); 
  } else if (keep == duplicate_keep_option::KEEP_LAST) {
      return thrust::copy_if(exec,
              first,
              last,
              thrust::counting_iterator<IndexType>(0),
              output, 
              [first, comp, n] __device__ (IndexType i) mutable {
              return (i == n || !comp(first[i], first[i+1]));
              });
  } else {
      return thrust::copy_if(exec,
              first,
              last,
              thrust::counting_iterator<IndexType>(0),
              output, 
              [first, comp, n] __device__ (IndexType i) mutable {
              return (i == 0 || !comp(first[i], first[i-1])) 
                  && (i == n || !comp(first[i], first[i+1]));
              });
  }
}

auto get_unique_ordered_indices(cudf::table_view const& keys,
                                cudf::column &unique_indices,
                                duplicate_keep_option const& keep,
                                bool const& nulls_are_equal = true,
                                rmm::mr::device_memory_resource* mr =
                                rmm::mr::get_default_resource(),
                                cudaStream_t stream=0)
{
  // sort only indices
  auto sorted_indices = sorted_order(keys,
                                     std::vector<order>{},
                                     std::vector<null_order>{},
                                     stream,
                                     mr);


  // extract unique indices 
  auto device_input_table = cudf::table_device_view::create(keys, stream);

  if(cudf::has_nulls(keys)) {
    auto comp = row_equality_comparator<true>(*device_input_table,
                                              *device_input_table,
                                               nulls_are_equal);
    auto result_end = unique_copy(rmm::exec_policy(stream)->on(stream),
        sorted_indices->view().begin<cudf::size_type>(),
        sorted_indices->view().end<cudf::size_type>(),
        unique_indices.mutable_view().begin<cudf::size_type>(),
        comp,
        keep);
  
    return cudf::detail::slice(unique_indices.view(), 0, thrust::distance(unique_indices.mutable_view().begin<cudf::size_type>(), result_end));
  } else {
    auto comp = row_equality_comparator<false>(*device_input_table,
                                               *device_input_table,
                                               nulls_are_equal);
    auto result_end = unique_copy(rmm::exec_policy(stream)->on(stream),
        sorted_indices->view().begin<cudf::size_type>(),
        sorted_indices->view().end<cudf::size_type>(),
        unique_indices.mutable_view().begin<cudf::size_type>(),
        comp,
        keep);
  
    return cudf::detail::slice(unique_indices.view(), 0, thrust::distance(unique_indices.mutable_view().begin<cudf::size_type>(), result_end));
  }
  
}

cudf::size_type unique_count(table_view const& keys,
                             bool const& nulls_are_equal = true,
                             rmm::mr::device_memory_resource* mr 
                                 = rmm::mr::get_default_resource(),
                             cudaStream_t stream=0)
{
  // sort only indices
  auto sorted_indices = sorted_order(keys,
                                     std::vector<order>{},
                                     std::vector<null_order>{},
                                     stream,
                                     mr);
  
  // count unique elements
  cudf::size_type* sorted_row_index = const_cast<cudf::size_type *>(sorted_indices->view().data<cudf::size_type>());
  auto device_input_table = cudf::table_device_view::create(keys, stream);

  if(cudf::has_nulls(keys)) {
    row_equality_comparator<true> comp (*device_input_table,
                                              *device_input_table,
                                              nulls_are_equal);
    return thrust::count_if(rmm::exec_policy(stream)->on(stream),
              thrust::counting_iterator<cudf::size_type>(0),
              thrust::counting_iterator<cudf::size_type>(keys.num_rows()),
              [sorted_row_index, comp]
              __device__ (cudf::size_type i) mutable {
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
              __device__ (cudf::size_type i) mutable {
              return (i == 0 || not comp(sorted_row_index[i], sorted_row_index[i-1]));
              });
  }
}

std::unique_ptr<experimental::table>
  drop_duplicates(table_view const& input,
                  table_view const& keys,
                  duplicate_keep_option const& keep,
                  bool const& nulls_are_equal,
                  rmm::mr::device_memory_resource* mr,
                  cudaStream_t stream)
{
  if (0 == input.num_rows() || 
      0 == input.num_columns() ||
      0 == keys.num_columns()
      ) {
      std::vector<std::unique_ptr<column>> out_columns(input.num_columns());
      std::transform(input.begin(), input.end(), out_columns.begin(),
                [&stream] (auto col_view){
                return detail::empty_like(col_view, stream);
                });

    return std::make_unique<experimental::table>(std::move(out_columns));
  }
  
  CUDF_EXPECTS( input.num_rows() == keys.num_rows(), "number of \
rows in input table should be equal to number of rows in key colums table");

  // The values will be filled into this column
  auto unique_indices = 
        cudf::make_numeric_column(data_type{INT32}, 
                                  keys.num_rows(), UNALLOCATED, stream, mr);
  // This is just slice of `unique_indices` but with different size as per the
  // keys has been processed in `get_unique_ordered_indices`
  auto unique_indices_view = 
      detail::get_unique_ordered_indices(keys, 
                                         *unique_indices,
                                         keep, nulls_are_equal);
 
  // run gather operation to establish new order
  return detail::gather(input, unique_indices_view, false, false, true, mr, stream);
}

template <typename T>
struct check_for_nan
{
    check_for_nan(cudf::column_device_view input) :_input{input}{}
  __device__
  bool operator()(size_type index)
  {
    return std::isnan(_input.data<T>()[index]) and _input.is_valid(index);
  }

protected:
  cudf::column_device_view _input;
};

cudf::size_type unique_count(column_view const& input,
                             bool const& ignore_nulls,
                             bool const& nan_as_null,
                             rmm::mr::device_memory_resource *mr,
                             cudaStream_t stream)
{
  if (0 == input.size() || input.null_count() == input.size()) {
    return 0;
  }

  cudf::size_type nrows = input.size();
 
  bool has_nans = false;
  // Check for Nans
  if (input.type().id() == FLOAT32 and input.has_nulls() and nan_as_null) {

      auto input_device_view = cudf::column_device_view::create(input, stream);
      auto device_view = *input_device_view;
      has_nans = thrust::any_of(rmm::exec_policy(stream)->on(stream), 
                                thrust::counting_iterator<cudf::size_type>(0),
                                thrust::counting_iterator<cudf::size_type>(nrows),
                                check_for_nan<float>(device_view));

  }
  else if (input.type().id() == FLOAT64 and input.has_nulls() and nan_as_null) {

      auto input_device_view = cudf::column_device_view::create(input, stream);
      auto device_view = *input_device_view;
      has_nans = thrust::any_of(rmm::exec_policy(stream)->on(stream),
                                thrust::counting_iterator<cudf::size_type>(0),
                                thrust::counting_iterator<cudf::size_type>(nrows),
                                check_for_nan<double>(device_view));
      
  }

  auto count = detail::unique_count(table_view{{input}}, true, mr, stream);


  // if nan is considered null and there are already null values
  if (nan_as_null and has_nans and input.has_nulls())
    --count;

  if(ignore_nulls and input.has_nulls())
    return --count;
  else
    return count;
}

}// namespace detail

std::unique_ptr<experimental::table>
  drop_duplicates(table_view const& input,
                  table_view const& keys,
                  duplicate_keep_option const& keep,
                  bool const& nulls_are_equal,
                  rmm::mr::device_memory_resource* mr) {

    return detail::drop_duplicates(input, keys, keep, nulls_are_equal, mr);
}

cudf::size_type unique_count(column_view const& input,
                             bool const& ignore_nulls,
                             bool const& nan_as_null,
                             rmm::mr::device_memory_resource *mr) {

    return detail::unique_count(input, ignore_nulls, nan_as_null, mr);
}

}// namespace experimental
}// namespace cudf

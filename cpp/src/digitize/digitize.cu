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
#include <cudf/digitize.hpp>
#include <cudf/detail/digitize.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/error.hpp>
#include <thrust/binary_search.h>

namespace cudf {

namespace {

template <bool has_nulls>
std::unique_ptr<column>
digitize_impl(column_view const& col, column_view const& bins, range_bound bound, null_order order,
              rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  auto output = cudf::make_numeric_column(data_type{INT32}, col.size(),
    cudf::UNALLOCATED, stream, mr);
  auto output_view = output->mutable_view();

  auto col_device_view = column_device_view::create(col, stream);
  auto bins_device_view = column_device_view::create(bins, stream);
  auto comparator = experimental::element_less_comparator<has_nulls>(
    *bins_device_view, *col_device_view, order);

  if (bound == range_bound::EXCLUSIVE) {
    thrust::upper_bound(rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<experimental::left_index>(0),
      thrust::make_counting_iterator<experimental::left_index>(bins.size()),
      thrust::make_counting_iterator<experimental::right_index>(0),
      thrust::make_counting_iterator<experimental::right_index>(col.size()),
      output_view.begin<int32_t>(), comparator);
  } else {
    thrust::lower_bound(rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<experimental::left_index>(0),
      thrust::make_counting_iterator<experimental::left_index>(bins.size()),
      thrust::make_counting_iterator<experimental::right_index>(0),
      thrust::make_counting_iterator<experimental::right_index>(col.size()),
      output_view.begin<int32_t>(), comparator);
  }

  return output;
}

}  // namespace

namespace detail {

std::unique_ptr<column>
digitize(column_view const& col, column_view const& bins, range_bound bound, null_order order,
         rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  CUDF_EXPECTS(col.type() == bins.type(), "Column type mismatch");

  if (col.has_nulls() || bins.has_nulls()) {
    return digitize_impl<true>(col, bins, bound, order, mr, stream);
  } else {
    return digitize_impl<false>(col, bins, bound, order, mr, stream);
  }
}

}  // namespace detail

std::unique_ptr<column>
digitize(column_view const& col, column_view const& bins, range_bound bound, null_order order,
         rmm::mr::device_memory_resource* mr)
{
  return detail::digitize(col, bins, bound, order, mr);
}

}  // namespace cudf

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
#include "digitize.hpp"
#include <cudf/digitize.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <utilities/error_utils.hpp>
#include <thrust/binary_search.h>

namespace cudf {

namespace {

// Initialize column with sequential values starting from 0
auto make_sequence(size_type num_rows, rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
  auto indices = cudf::make_numeric_column(data_type{INT32}, num_rows, mask_state::UNALLOCATED, stream, mr);
  auto view = indices->mutable_view();
  thrust::sequence(rmm::exec_policy(stream)->on(stream), view.begin<int32_t>(), view.end<int32_t>());
  return indices;
}

// TODO this is a specialization of row_lexicographic_comparator for single columns
template <bool has_nulls = true>
class less_equal_wrapper {
 public:
  less_equal_wrapper(column_device_view lhs, column_device_view rhs,
                                        null_order null_precedence = null_order::BEFORE)
      : _comparator{lhs, rhs, null_precedence}, _dtype(lhs.type()) {}

  __device__ bool operator()(size_type lhs_index, size_type rhs_index) const {
    return experimental::type_dispatcher(_dtype, _comparator, lhs_index, rhs_index)
      != experimental::weak_ordering::GREATER;
  }

 private:
  experimental::element_relational_comparator<has_nulls> _comparator;
  data_type _dtype;
};

struct nullable_binary_search_bound {
  null_order _null_precedence = null_order::BEFORE;

  template<typename T>
  auto operator()(column_view const& col, column_view const& bins, bool upper_bound,
                  rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    // TODO allocate null mask and copy contents from col?
    auto output = cudf::make_numeric_column(data_type{INT32}, col.size(),
      cudf::UNALLOCATED, stream, mr);
    auto output_view = output->mutable_view();

    // Allocate index columns to be used with element_relation_comparator
    auto col_indices = make_sequence(col.size(), mr, stream);
    auto col_indices_view = col_indices->view();
    auto bins_indices = make_sequence(bins.size(), mr, stream);
    auto bins_indices_view = bins_indices->view();

    auto col_device_view = column_device_view::create(col, stream);
    auto bins_device_view = column_device_view::create(bins, stream);
    auto comparator = less_equal_wrapper<true>(
        *bins_device_view, *col_device_view, _null_precedence);

    if (upper_bound) {
      thrust::upper_bound(rmm::exec_policy(stream)->on(stream),
        bins_indices_view.begin<int32_t>(), bins_indices_view.end<int32_t>(),
        col_indices_view.begin<int32_t>(), col_indices_view.end<int32_t>(),
        output_view.begin<int32_t>(), comparator);
    } else {
      thrust::lower_bound(rmm::exec_policy(stream)->on(stream),
        bins_indices_view.begin<int32_t>(), bins_indices_view.end<int32_t>(),
        col_indices_view.begin<int32_t>(), col_indices_view.end<int32_t>(),
        output_view.begin<int32_t>(), comparator);
    }

    return output;
  }
};

struct binary_search_bound {
  template<typename T>
  auto operator()(column_view const& col, column_view const& bins, bool upper_bound,
                  rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    auto output = cudf::make_numeric_column(data_type{INT32}, col.size(),
      cudf::UNALLOCATED, stream, mr);
    auto output_view = output->mutable_view();

    if (upper_bound) {
      thrust::upper_bound(rmm::exec_policy(stream)->on(stream),
        bins.begin<T>(), bins.end<T>(), col.begin<T>(), col.end<T>(),
        output_view.begin<int32_t>(), thrust::less_equal<T>());
    } else {
      thrust::lower_bound(rmm::exec_policy(stream)->on(stream),
        bins.begin<T>(), bins.end<T>(), col.begin<T>(), col.end<T>(),
        output_view.begin<int32_t>(), thrust::less_equal<T>());
    }

    return output;
  }
};

}  // namespace

namespace detail {

std::unique_ptr<column>
digitize(column_view const& col, column_view const& bins, bool right,
         rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  auto const dtype = col.type();
  CUDF_EXPECTS(dtype == bins.type(), "Column type mismatch");

  // TODO should this make use of the comparable type traits?
  CUDF_EXPECTS(is_numeric(dtype) || is_timestamp(dtype), "Type must be numeric or timestamp");

  CUDF_EXPECTS(0 == bins.null_count(), "Bins column must not have nulls");

  if (col.null_count() > 0) {
    return experimental::type_dispatcher(dtype, nullable_binary_search_bound{null_order::BEFORE},
      col, bins, right, mr, stream);
  } else {
    return experimental::type_dispatcher(dtype, binary_search_bound{},
      col, bins, right, mr, stream);
  }
}

}  // namespace detail

std::unique_ptr<column>
digitize(column_view const& col, column_view const& bins, bool right,
         rmm::mr::device_memory_resource* mr)
{
  return detail::digitize(col, bins, right, mr);
}

}  // namespace cudf

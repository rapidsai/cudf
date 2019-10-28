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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/legacy/copying.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>

namespace cudf {
namespace experimental {

namespace {

template <typename DataIterator, typename ValuesIterator,
          typename OutputIterator, typename Comparator>
void launch_search(DataIterator it_data,
                   ValuesIterator it_vals,
                   size_type data_size,
                   size_type values_size,
                   OutputIterator it_output,
                   Comparator comp,
                   bool find_first,
                   cudaStream_t stream)
{
  if (find_first) {
    thrust::lower_bound(rmm::exec_policy(stream)->on(stream),
                        it_data, it_data + data_size,
                        it_vals, it_vals + values_size,
                        it_output, comp);
  }
  else {
    thrust::upper_bound(rmm::exec_policy(stream)->on(stream),
                        it_data, it_data + data_size,
                        it_vals, it_vals + values_size,
                        it_output, comp);
  }
}

} // namespace

namespace detail {

std::unique_ptr<column> search_ordered(table_view const& t,
                                       table_view const& values,
                                       bool find_first,
                                       std::vector<order> const& column_order,
                                       null_order null_precedence,
                                       rmm::mr::device_memory_resource *mr,
                                       cudaStream_t stream = 0)
{
  // Allocate result column
  std::unique_ptr<column> result = make_numeric_column(data_type{INT32}, values.num_rows(),
                                                       mask_state::UNALLOCATED, stream, mr);

  mutable_column_view result_view = result.get()->mutable_view();

  // Handle empty inputs
  if (t.num_rows() == 0) {
    CUDA_TRY(cudaMemset(result_view.data<int32_t>(), 0, values.num_rows() * sizeof(int32_t)));
    return result;
  }

  if (not column_order.empty()) {
    CUDF_EXPECTS(
        static_cast<std::size_t>(t.num_columns()) == column_order.size(),
        "Mismatch between number of columns and column order.");
  }

  auto d_t      = table_device_view::create(t, stream);
  auto d_values = table_device_view::create(values, stream);
  auto count_it = thrust::make_counting_iterator<int32_t>(0);

  //  Need an order*
  rmm::device_vector<order> d_column_order(column_order.begin(), column_order.end());

  if (has_nulls(t)) {
    auto ineq_op = (find_first)
                 ? row_lexicographic_comparator<true>(*d_t, *d_values, null_precedence, d_column_order.data().get())
                 : row_lexicographic_comparator<true>(*d_values, *d_t, null_precedence, d_column_order.data().get());

    launch_search(count_it, count_it, t.num_rows(), values.num_rows(),
                  result_view.data<int32_t>(), ineq_op, find_first, stream);
  } else {
    auto ineq_op = (find_first)
                 ? row_lexicographic_comparator<false>(*d_t, *d_values, null_precedence, d_column_order.data().get())
                 : row_lexicographic_comparator<false>(*d_values, *d_t, null_precedence, d_column_order.data().get());

    launch_search(count_it, count_it, t.num_rows(), values.num_rows(),
                  result_view.data<int32_t>(), ineq_op, find_first, stream);
  }

  return result;
}

template <bool nullable = true>
struct compare_with_value{
  compare_with_value(table_device_view t, table_device_view val, bool nulls_are_equal = true)
    : compare(t, val, nulls_are_equal) {}

  __device__ bool operator()(int32_t i){
    return compare(i, 0);
  }
  row_equality_comparator<nullable> compare;
};

bool contains(column_view const& col,
              gdf_scalar const& value,
              cudaStream_t stream = 0)
{
  // TODO: Rework for cudf::exp::scalar
  //       Any reference to value is suspect

  // TODO:  Not sure how to do this!!!
  //CUDF_EXPECTS(col.type() == value.dtype, "DTYPE mismatch");

  // No element to compare against
  if (col.size() == 0) {
    return false;
  }

  // If value is invalid and there are any nulls, return true
  if (value.is_valid == false){
    return col.has_nulls();
  }

  std::unique_ptr<column> scalar_as_column = make_numeric_column(col.type(), 1);

  CUDA_TRY(cudaMemcpyAsync(scalar_as_column.get()->mutable_view().data<int32_t>(),
                           &value.data,
                           //cudf::size_of(value.dtype),
                           sizeof(int32_t),
                           cudaMemcpyHostToDevice, stream));

  auto d_t = cudf::table_device_view::create(table_view{{col}});
  auto sss = scalar_as_column.get()->view();
  //auto d_value = cudf::table_device_view::create(table_view{{scalar_as_column.get()->view()}});
  auto d_value = cudf::table_device_view::create(table_view{{sss}});

  //  TODO:  What is this type?  above code assumes it is
  //     a counting iterator templated to gdf_index_type,
  //     but don't we have to specify this if we want that?
  //     Otherwise this should a counting iterator to int.
  //
  auto data_it = thrust::make_counting_iterator(0);

  if (col.has_nulls()) {
    auto eq_op = compare_with_value<true>(*d_t, *d_value, true);

    return thrust::any_of(rmm::exec_policy(stream)->on(stream),
                          data_it, data_it + col.size(),
                          eq_op);
  }
  else {
    auto eq_op = compare_with_value<false>(*d_t, *d_value, true);

    return thrust::any_of(rmm::exec_policy(stream)->on(stream),
                          data_it, data_it + col.size(),
                          eq_op);
  }
}
} // namespace detail

std::unique_ptr<column> lower_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    null_order null_precedence,
                                    rmm::mr::device_memory_resource *mr)
{
  return detail::search_ordered(t, values, true, column_order, null_precedence, mr);
}

std::unique_ptr<column> upper_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    null_order null_precedence,
                                    rmm::mr::device_memory_resource *mr)
{
  return detail::search_ordered(t, values, false, column_order, null_precedence, mr);
}

bool contains(column const& col, gdf_scalar const& value)
{
    return detail::contains(col, value);
}

} // namespace exp
} // namespace cudf

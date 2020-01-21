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

#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <cudf/utilities/legacy/wrapper_types.hpp>
#include <utilities/legacy/column_utils.hpp>
#include <io/utilities/wrapper_utils.hpp>

#include <cudf/legacy/search.hpp>
#include <cudf/legacy/copying.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

namespace cudf {

namespace {

template <typename DataIterator, typename ValuesIterator, typename Comparator>
void launch_search(DataIterator it_data,
                    ValuesIterator it_vals,
                    cudf::size_type data_size,
                    cudf::size_type values_size,
                    void* output,
                    Comparator comp,
                    bool find_first,
                    cudaStream_t stream)
{
  if (find_first) {
    thrust::lower_bound(rmm::exec_policy(stream)->on(stream),
                        it_data, it_data + data_size,
                        it_vals, it_vals + values_size,
                        static_cast<cudf::size_type*>(output),
                        comp);
  }
  else {
    thrust::upper_bound(rmm::exec_policy(stream)->on(stream),
                        it_data, it_data + data_size,
                        it_vals, it_vals + values_size,
                        static_cast<cudf::size_type*>(output),
                        comp);
  }
}

} // namespace

namespace detail {

gdf_column search_ordered(table const& t,
                          table const& values,
                          bool find_first,
                          std::vector<bool> const& desc_flags,
                          bool nulls_as_largest,
                          cudaStream_t stream = 0)
{
  // Allocate result column
  gdf_column result_like{};
  result_like.dtype = GDF_INT32;
  result_like.size = values.num_rows();
  result_like.data = values.get_column(0)->data;
  auto result = allocate_like(result_like);

  // Handle empty inputs
  if (t.num_rows() == 0) {
    CUDA_TRY(cudaMemset(result.data, 0, values.num_rows()));
    if (is_nullable(result)) {
      CUDA_TRY(cudaMemset(result.valid, 0, values.num_rows()));
    }
  }

  auto d_t      = device_table::create(t, stream);
  auto d_values = device_table::create(values, stream);
  auto count_it = thrust::make_counting_iterator(0);

  rmm::device_vector<int8_t> dv_desc_flags(desc_flags);
  auto d_desc_flags = dv_desc_flags.data().get();
  
  if ( has_nulls(t) ) {
    auto ineq_op = (find_first)
                 ? row_inequality_comparator<true>(*d_t, *d_values, !nulls_as_largest, d_desc_flags)
                 : row_inequality_comparator<true>(*d_values, *d_t, !nulls_as_largest, d_desc_flags);

    launch_search(count_it, count_it, t.num_rows(), values.num_rows(), result.data,
                  ineq_op, find_first, stream);
  }
  else {
    auto ineq_op = (find_first)
                 ? row_inequality_comparator<false>(*d_t, *d_values, !nulls_as_largest, d_desc_flags)
                 : row_inequality_comparator<false>(*d_values, *d_t, !nulls_as_largest, d_desc_flags);

    launch_search(count_it, count_it, t.num_rows(), values.num_rows(), result.data,
                  ineq_op, find_first, stream);
  }

  return result;
}

template <bool nullable = true>
struct compare_with_value{

            compare_with_value(device_table t, device_table val, bool nulls_are_equal = true)
                : compare(t, val, nulls_are_equal) {}

            __device__ bool operator()(cudf::size_type i){
               return compare(i, 0);
            }
            row_equality_comparator<nullable> compare;
};

bool contains(gdf_column const& column,
              gdf_scalar const& value,
              cudaStream_t stream = 0)
{
  CUDF_EXPECTS(column.dtype == value.dtype, "DTYPE mismatch");

  // No element to compare against
  if (column.size == 0) {
      return false;
  }

  if (value.is_valid == false){
      return cudf::has_nulls(column);
  }

  // Create column with scalar's data
  gdf_column_wrapper val (1, value.dtype, gdf_dtype_extra_info{}, "");
  RMM_TRY(RMM_ALLOC(&val.get()->data, cudf::size_of(value.dtype), stream));
  CUDA_TRY(cudaMemcpyAsync(val.get()->data, (void*) &value.data,
                  cudf::size_of(value.dtype), cudaMemcpyHostToDevice, stream));

  gdf_column* tmp_column = const_cast<gdf_column *> (&column);
  gdf_column* tmp_value = val.get();

  // Creating a single column device table
  auto d_t = device_table::create(1, &tmp_column, stream);
  auto d_value = device_table::create(1, &tmp_value, stream);
  auto data_it = thrust::make_counting_iterator(0);

  if (cudf::has_nulls(column)) {
    auto eq_op = compare_with_value<true>(*d_t, *d_value, true);

    return thrust::any_of(rmm::exec_policy(stream)->on(stream),
                          data_it, data_it + column.size,
                          eq_op);
  }
  else {
    auto eq_op = compare_with_value<false>(*d_t, *d_value, true);

    return thrust::any_of(rmm::exec_policy(stream)->on(stream),
                          data_it, data_it + column.size,
                          eq_op);
  }
}
} // namespace detail

gdf_column lower_bound(table const& t,
                       table const& values,
                       std::vector<bool> const& desc_flags,
                       bool nulls_as_largest)
{
  return detail::search_ordered(t, values, true, desc_flags, nulls_as_largest);
}

gdf_column upper_bound(table const& t,
                       table const& values,
                       std::vector<bool> const& desc_flags,
                       bool nulls_as_largest)
{
  return detail::search_ordered(t, values, false, desc_flags, nulls_as_largest);
}

bool contains(gdf_column const& column, gdf_scalar const& value)
{
    return detail::contains(column, value);
}
} // namespace cudf

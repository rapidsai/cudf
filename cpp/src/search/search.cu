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

#include <iterator/iterator.cuh>
#include <table/device_table.cuh>
#include <table/device_table_row_operators.cuh>
#include <utilities/wrapper_types.hpp>
#include <utilities/column_utils.hpp>

#include <cudf/search.hpp>
#include <cudf/copying.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

namespace cudf {

namespace {

template <typename DataIterator, typename ValuesIterator, typename Comparator>
void launch_search(DataIterator it_data,
                    ValuesIterator it_vals,
                    gdf_size_type data_size,
                    gdf_size_type values_size,
                    void* output,
                    Comparator comp,
                    bool find_first,
                    cudaStream_t stream)
{
  if (find_first) {
    thrust::lower_bound(rmm::exec_policy(stream)->on(stream),
                        it_data, it_data + data_size,
                        it_vals, it_vals + values_size,
                        static_cast<gdf_index_type*>(output),
                        comp);
  }
  else {
    thrust::upper_bound(rmm::exec_policy(stream)->on(stream),
                        it_data, it_data + data_size,
                        it_vals, it_vals + values_size,
                        static_cast<gdf_index_type*>(output),
                        comp);
  }
}

} // namespace

namespace detail {

gdf_column search_ordered(table const& t,
                          table const& values,
                          bool find_first,
                          std::vector<bool>& desc_flags,
                          bool nulls_as_largest,
                          cudaStream_t stream = 0)
{
  // TODO: validate input table and values
  // TODO: allow empty input

  // Allocate result column
  gdf_column result_like{};
  result_like.dtype = GDF_INT32;
  result_like.size = values.num_rows();
  result_like.data = values.get_column(0)->data;
  // TODO: let result have nulls? this could be used for records not found
  auto result = allocate_like(result_like);

  auto d_t      = device_table::create(t, stream);
  auto d_values = device_table::create(values, stream);
  auto count_it = thrust::make_counting_iterator(0);

  thrust::device_vector<int8_t, rmm_allocator<int8_t>> dv_desc_flags(desc_flags);
  auto d_desc_flags = thrust::raw_pointer_cast(dv_desc_flags.data());
  
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

gdf_column search_ordered(gdf_column const& column,
                          gdf_column const& values,
                          bool find_first,
                          bool ascending,
                          bool nulls_as_largest,
                          cudaStream_t stream = 0)
{
  const table t{const_cast<gdf_column*>(&column)};
  const table val{const_cast<gdf_column*>(&values)};
  std::vector<bool> desc_flags{!ascending};

  return search_ordered(t, val, find_first, desc_flags, nulls_as_largest, stream);
}

} // namespace detail

gdf_column lower_bound(gdf_column const& column,
                       gdf_column const& values,
                       bool ascending,
                       bool nulls_as_largest)
{
  return detail::search_ordered(column, values, true, ascending, nulls_as_largest);
}

gdf_column upper_bound(gdf_column const& column,
                       gdf_column const& values,
                       bool ascending,
                       bool nulls_as_largest)
{
  return detail::search_ordered(column, values, false, ascending, nulls_as_largest);
}

gdf_column lower_bound(table const& t,
                       table const& values,
                       std::vector<bool>& desc_flags,
                       bool nulls_as_largest)
{
  return detail::search_ordered(t, values, true, desc_flags, nulls_as_largest);
}

gdf_column upper_bound(table const& t,
                       table const& values,
                       std::vector<bool>& desc_flags,
                       bool nulls_as_largest)
{
  return detail::search_ordered(t, values, false, desc_flags, nulls_as_largest);
}

} // namespace cudf

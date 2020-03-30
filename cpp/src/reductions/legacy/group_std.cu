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

#include "group_reductions.hpp"

#include <bitmask/legacy/bit_mask.cuh>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <groupby/sort/legacy/sort_helper.hpp>

#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>

#include <cmath>

namespace {

struct var_functor {
  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, void >
  operator()(gdf_column const& values,
             rmm::device_vector<cudf::size_type> const& group_labels,
             rmm::device_vector<cudf::size_type> const& group_sizes,
             gdf_column * result,
             cudf::size_type ddof,
             bool is_std,
             cudaStream_t stream)
  {
    auto values_data = static_cast<const T*>(values.data);
    auto result_data = static_cast<double *>(result->data);
    auto values_valid = reinterpret_cast<const bit_mask::bit_mask_t*>(values.valid);
    auto result_valid = reinterpret_cast<bit_mask::bit_mask_t*>(result->valid);
    const cudf::size_type* d_group_labels = group_labels.data().get();
    const cudf::size_type* d_group_sizes = group_sizes.data().get();
    
    // Calculate sum
    // TODO: replace with mean function call when that gets an internal API
    rmm::device_vector<T> sums(group_sizes.size());

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          group_labels.begin(), group_labels.end(),
                          thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            [=] __device__ (cudf::size_type i) -> T {
                              return (values_valid and not bit_mask::is_valid(values_valid, i))
                                     ? 0 : values_data[i];
                            }),
                          thrust::make_discard_iterator(),
                          sums.begin());

    // TODO: use target_type for sums and result_data
    T* d_sums = sums.data().get();

    auto values_it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [=] __device__ (cudf::size_type i) {
        if (values_valid and not bit_mask::is_valid(values_valid, i))
          return 0.0;
        
        double x = values_data[i];
        cudf::size_type group_idx = d_group_labels[i];
        cudf::size_type group_size = d_group_sizes[group_idx];
        
        // prevent divide by zero error
        if (group_size == 0 or group_size - ddof <= 0)
          return 0.0;

        double mean = static_cast<double>(d_sums[group_idx])/group_size;
        return (x - mean) * (x - mean) / (group_size - ddof);
      }
    );

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          group_labels.begin(), group_labels.end(), values_it, 
                          thrust::make_discard_iterator(),
                          result_data);

    // set nulls
    if (result_valid) {
      thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator(0), group_sizes.size(),
        [=] __device__ (cudf::size_type i){
          cudf::size_type group_size = d_group_sizes[i];
          if (group_size == 0 or group_size - ddof <= 0)
            bit_mask::clear_bit_safe(result_valid, i);
          else
            bit_mask::set_bit_safe(result_valid, i);
        });
      set_null_count(*result);
    }

    // if std, do a sqrt
    if (is_std) {
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        result_data, result_data + group_sizes.size(),
                        result_data,
        [] __device__ (double data) { return sqrt(data); });
    }
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic<T>::value, void >
  operator()(Args&&... args) {
    CUDF_FAIL("Only numeric types are supported in variance");
  }
};

} // namespace anonymous


namespace cudf {
namespace detail {

void group_var(gdf_column const& values,
               rmm::device_vector<size_type> const& group_labels,
               rmm::device_vector<size_type> const& group_sizes,
               gdf_column * result,
               size_type ddof,
               cudaStream_t stream)
{
  type_dispatcher(values.dtype, var_functor{},
    values, group_labels, group_sizes, result, ddof, false, stream);
}

void group_std(gdf_column const& values,
               rmm::device_vector<size_type> const& group_labels,
               rmm::device_vector<size_type> const& group_sizes,
               gdf_column * result,
               size_type ddof,
               cudaStream_t stream)
{
  type_dispatcher(values.dtype, var_functor{},
    values, group_labels, group_sizes, result, ddof, true, stream);
}

std::pair<cudf::table, cudf::table>
group_var_std(cudf::table const& keys,
              cudf::table const& values,
              cudf::size_type ddof,
              bool is_std)
{
  groupby::sort::detail::helper gb_obj(keys);
  auto group_labels = gb_obj.group_labels();

  cudf::table result_table(gb_obj.num_groups(),
                           std::vector<gdf_dtype>(values.num_columns(), GDF_FLOAT64),
                           std::vector<gdf_dtype_extra_info>(values.num_columns()),
                           true);

  for (cudf::size_type i = 0; i < values.num_columns(); i++)
  {
    gdf_column sorted_values;
    rmm::device_vector<cudf::size_type> group_sizes;
    std::tie(sorted_values, group_sizes) =
      gb_obj.sort_values(*(values.get_column(i)));

    gdf_column* result_col = result_table.get_column(i);

    if (is_std) {
      detail::group_std(sorted_values, group_labels, group_sizes,
                        result_col, ddof);
    } else {
      detail::group_var(sorted_values, group_labels, group_sizes,
                        result_col, ddof);
    }

    gdf_column_free(&sorted_values);
  }

  return std::make_pair(gb_obj.unique_keys(), std::move(result_table));
}

} // namespace detail

std::pair<cudf::table, cudf::table>
group_std(cudf::table const& keys,
          cudf::table const& values,
          cudf::size_type ddof)
{
  return detail::group_var_std(keys, values, ddof, true);
}

std::pair<cudf::table, cudf::table>
group_var(cudf::table const& keys,
          cudf::table const& values,
          cudf::size_type ddof)
{
  return detail::group_var_std(keys, values, ddof, false);
}

} // namespace cudf

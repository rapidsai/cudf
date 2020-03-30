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

#include <quantiles/legacy/group_quantiles.hpp>

#include <utilities/legacy/cuda_utils.hpp>
#include <quantiles/legacy/quantiles_util.hpp>
#include <groupby/sort/legacy/sort_helper.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/legacy/column.hpp>

#include <rmm/rmm.h>

#include <thrust/for_each.h>

namespace cudf {

namespace {

struct quantiles_functor {

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, void >
  operator()(gdf_column const& values_col,
             rmm::device_vector<cudf::size_type> const& group_offsets,
             rmm::device_vector<cudf::size_type> const& group_sizes,
             gdf_column* result_col, rmm::device_vector<double> const& quantile,
             cudf::interpolation interpolation, cudaStream_t stream = 0)
  {
    // prepare args to be used by lambda below
    auto result = static_cast<double*>(result_col->data);
    auto values = static_cast<T*>(values_col.data);
    auto group_id = group_offsets.data().get();
    auto group_size = group_sizes.data().get();
    auto d_quantiles = quantile.data().get();
    auto num_quantiles = quantile.size();

    // For each group, calculate quantile
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator(0),
      group_offsets.size(),
      [=] __device__ (cudf::size_type i) {
        cudf::size_type segment_size = group_size[i];

        auto value = values + group_id[i];
        thrust::transform(thrust::seq, d_quantiles, d_quantiles + num_quantiles,
                          result + i * num_quantiles,
                          [=](auto q) { 
                            return detail::select_quantile(value,
                                                           segment_size, 
                                                           q, 
                                                           interpolation); 
                          });
      }
    );
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic<T>::value, void >
  operator()(Args&&... args) {
    CUDF_FAIL("Only arithmetic types are supported in quantiles");
  }
};

} // namespace anonymous

namespace detail {

void group_quantiles(gdf_column const& values,
                     rmm::device_vector<cudf::size_type> const& group_offsets,
                     rmm::device_vector<cudf::size_type> const& group_sizes,
                     gdf_column * result,
                     std::vector<double> const& quantiles,
                     cudf::interpolation interpolation,
                     cudaStream_t stream)
{
  rmm::device_vector<double> dv_quantiles(quantiles);

  type_dispatcher(values.dtype, quantiles_functor{},
                  values, group_offsets, group_sizes, result,
                  dv_quantiles, interpolation, stream);
}

void group_medians(gdf_column const& values,
                   rmm::device_vector<cudf::size_type> const& group_offsets,
                   rmm::device_vector<cudf::size_type> const& group_sizes,
                   gdf_column * result,
                   cudaStream_t stream)
{
  std::vector<double> quantiles{0.5};
  cudf::interpolation interpolation = cudf::interpolation::LINEAR;

  rmm::device_vector<double> dv_quantiles(quantiles);

  type_dispatcher(values.dtype, quantiles_functor{},
                  values, group_offsets, group_sizes, result,
                  dv_quantiles, interpolation, stream);
}

} // namespace detail

// TODO: add optional check for is_sorted. Use context.flag_sorted
std::pair<cudf::table, cudf::table>
group_quantiles(cudf::table const& keys,
                cudf::table const& values,
                std::vector<double> const& quantiles,
                cudf::interpolation interpolation,
                bool include_nulls)
{
  groupby::sort::detail::helper gb_obj(keys, include_nulls);
  auto group_offsets = gb_obj.group_offsets();

  cudf::table result_table(gb_obj.num_groups() * quantiles.size(),
                           std::vector<gdf_dtype>(values.num_columns(), GDF_FLOAT64),
                           std::vector<gdf_dtype_extra_info>(values.num_columns()));

  for (cudf::size_type i = 0; i < values.num_columns(); i++)
  {
    gdf_column sorted_values;
    rmm::device_vector<cudf::size_type> group_sizes;
    std::tie(sorted_values, group_sizes) =
      gb_obj.sort_values(*(values.get_column(i)));

    gdf_column* result_col = result_table.get_column(i);

    detail::group_quantiles(sorted_values, group_offsets, group_sizes, 
                            result_col, quantiles, interpolation);

    gdf_column_free(&sorted_values);
  }

  return std::make_pair(gb_obj.unique_keys(), std::move(result_table));
}
    
} // namespace cudf

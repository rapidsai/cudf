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

#include "groupby.hpp"

#include <utilities/cuda_utils.hpp>
#include <quantiles/quantiles.hpp>
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
             rmm::device_vector<gdf_size_type> const& group_indices,
             rmm::device_vector<gdf_size_type> const& group_sizes,
             gdf_column& result_col, rmm::device_vector<double> const& quantile,
             quantile_method interpolation, cudaStream_t stream)
  {
    // prepare args to be used by lambda below
    auto result = reinterpret_cast<double*>(result_col.data);
    auto values = reinterpret_cast<T*>(values_col.data);
    auto group_id = group_indices.data().get();
    auto group_size = group_sizes.data().get();
    auto d_quantiles = quantile.data().get();
    auto num_quantiles = quantile.size();

    // For each group, calculate quantile
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator(0),
      group_indices.size(),
      [=] __device__ (gdf_size_type i) {
        gdf_size_type segment_size = group_size[i];

        for (gdf_size_type j = 0; j < num_quantiles; j++) {
          gdf_size_type k = i * num_quantiles + j;
          result[k] = detail::select_quantile(values + group_id[i], segment_size,
                                              d_quantiles[j], interpolation);
        }
      }
    );
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic<T>::value, void >
  operator()(Args&&... args) {
    CUDF_FAIL("Only arithmetic types are supported in quantile");
  }
};

} // namespace anonymous


// TODO: add optional check for is_sorted. Use context.flag_sorted
std::pair<cudf::table, cudf::table>
group_quantiles(cudf::table const& keys,
                cudf::table const& values,
                std::vector<double> const& quantiles,
                quantile_method interpolation,
                bool include_nulls)
{
  auto gb_obj = detail::groupby(keys, include_nulls);
  auto group_indices = gb_obj.group_indices();

  rmm::device_vector<double> dv_quantiles(quantiles);

  cudf::table result_table(gb_obj.num_groups() * quantiles.size(),
                           std::vector<gdf_dtype>(values.num_columns(), GDF_FLOAT64),
                           std::vector<gdf_dtype_extra_info>(values.num_columns()));

  util::cuda::scoped_stream stream{};

  for (gdf_size_type i = 0; i < values.num_columns(); i++)
  {
    gdf_column sorted_values;
    rmm::device_vector<gdf_size_type> group_sizes;
    std::tie(sorted_values, group_sizes) =
      gb_obj.sort_values(*(values.get_column(i)));

    auto& result_col = *(result_table.get_column(i));

    type_dispatcher(sorted_values.dtype, quantiles_functor{},
                    sorted_values, group_indices, group_sizes, result_col,
                    dv_quantiles, interpolation, stream);

    gdf_column_free(&sorted_values);
  }

  return std::make_pair(gb_obj.unique_keys(), result_table);
}
    
} // namespace cudf



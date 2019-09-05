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
 
//  #include <copying/copy_range.cuh>

namespace cudf {

namespace detail {

void shift(
  gdf_column *out_column,
  gdf_column const &in_column,
  gdf_index_type period,
  gdf_scalar const *fill_value
)
{
  if (period >= 0)
  {
    detail::copy_range(out_column, detail::column_range_factory{in_column, 0}, period, out_column->size);

    if (fill_value != nullptr)
    {
      detail::copy_range(out_column, detail::scalar_factory{*fill_value}, 0, period);
    }
  }
  else
  {
    auto mid = out_column->size + period;
    detail::copy_range(out_column, detail::column_range_factory{in_column, -period}, 0, mid);

    if (fill_value != nullptr)
    {
      detail::copy_range(out_column, detail::scalar_factory{*fill_value}, mid, out_column->size);
    }
  }
}
    
}; // namespace: detail

}; // namespace: cudf

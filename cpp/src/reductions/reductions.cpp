/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/sorting.hpp>

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace cudf {
namespace detail {
struct reduce_dispatch_functor {
  column_view const col;
  data_type output_dtype;
  rmm::mr::device_memory_resource *mr;
  cudaStream_t stream;

  reduce_dispatch_functor(column_view const &col,
                          data_type output_dtype,
                          rmm::mr::device_memory_resource *mr,
                          cudaStream_t stream)
    : col(col), output_dtype(output_dtype), mr(mr), stream(stream)
  {
  }

  template <aggregation::Kind k>
  std::unique_ptr<scalar> operator()(std::unique_ptr<aggregation> const &agg)
  {
    switch (k) {
      case aggregation::SUM: return reduction::sum(col, output_dtype, mr, stream); break;
      case aggregation::PRODUCT: return reduction::product(col, output_dtype, mr, stream); break;
      case aggregation::MIN: return reduction::min(col, output_dtype, mr, stream); break;
      case aggregation::MAX: return reduction::max(col, output_dtype, mr, stream); break;
      case aggregation::ANY: return reduction::any(col, output_dtype, mr, stream); break;
      case aggregation::ALL: return reduction::all(col, output_dtype, mr, stream); break;
      case aggregation::SUM_OF_SQUARES:
        return reduction::sum_of_squares(col, output_dtype, mr, stream);
        break;
      case aggregation::MEAN: return reduction::mean(col, output_dtype, mr, stream); break;
      case aggregation::VARIANCE: {
        auto var_agg = static_cast<std_var_aggregation const *>(agg.get());
        return reduction::variance(col, output_dtype, var_agg->_ddof, mr, stream);
      } break;
      case aggregation::STD: {
        auto var_agg = static_cast<std_var_aggregation const *>(agg.get());
        return reduction::standard_deviation(col, output_dtype, var_agg->_ddof, mr, stream);
      } break;
      case aggregation::MEDIAN: {
        auto sorted_indices       = sorted_order(table_view{{col}}, {}, {null_order::AFTER}, mr);
        auto valid_sorted_indices = split(*sorted_indices, {col.size() - col.null_count()})[0];
        auto col_ptr = quantile(col, {0.5}, interpolation::LINEAR, valid_sorted_indices, true, mr);
        return get_element(*col_ptr, 0, mr);
      } break;
      case aggregation::QUANTILE: {
        auto quantile_agg = static_cast<quantile_aggregation const *>(agg.get());
        CUDF_EXPECTS(quantile_agg->_quantiles.size() == 1,
                     "Reduction quantile accepts only one quantile value");
        auto sorted_indices       = sorted_order(table_view{{col}}, {}, {null_order::AFTER}, mr);
        auto valid_sorted_indices = split(*sorted_indices, {col.size() - col.null_count()})[0];
        auto col_ptr              = quantile(col,
                                quantile_agg->_quantiles,
                                quantile_agg->_interpolation,
                                valid_sorted_indices,
                                true,
                                mr);
        return get_element(*col_ptr, 0, mr);
      } break;
      case aggregation::NUNIQUE: {
        auto nunique_agg = static_cast<nunique_aggregation const *>(agg.get());
        return make_fixed_width_scalar(
          detail::distinct_count(
            col, nunique_agg->_null_handling, nan_policy::NAN_IS_VALID, stream),
          stream,
          mr);
      } break;
      default: CUDF_FAIL("Unsupported reduction operator");
    }
  }
};

std::unique_ptr<scalar> reduce(
  column_view const &col,
  std::unique_ptr<aggregation> const &agg,
  data_type output_dtype,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  std::unique_ptr<scalar> result = make_default_constructed_scalar(output_dtype);
  result->set_valid(false, stream);

  // check if input column is empty
  if (col.size() <= col.null_count()) return result;

  result =
    aggregation_dispatcher(agg->kind, reduce_dispatch_functor{col, output_dtype, mr, stream}, agg);
  return result;
}
}  // namespace detail

std::unique_ptr<scalar> reduce(column_view const &col,
                               std::unique_ptr<aggregation> const &agg,
                               data_type output_dtype,
                               rmm::mr::device_memory_resource *mr)
{
  CUDF_FUNC_RANGE();
  return detail::reduce(col, agg, output_dtype, mr);
}

}  // namespace cudf

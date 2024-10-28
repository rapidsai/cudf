/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/quantiles.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/detail/histogram.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <utility>

namespace cudf {
namespace reduction {
namespace detail {
struct reduce_dispatch_functor {
  column_view const col;
  data_type output_dtype;
  std::optional<std::reference_wrapper<scalar const>> init;
  rmm::device_async_resource_ref mr;
  rmm::cuda_stream_view stream;

  reduce_dispatch_functor(column_view col,
                          data_type output_dtype,
                          std::optional<std::reference_wrapper<scalar const>> init,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
    : col(std::move(col)), output_dtype(output_dtype), init(init), mr(mr), stream(stream)
  {
  }

  template <aggregation::Kind k>
  std::unique_ptr<scalar> operator()(reduce_aggregation const& agg)
  {
    switch (k) {
      case aggregation::SUM: return sum(col, output_dtype, init, stream, mr);
      case aggregation::PRODUCT: return product(col, output_dtype, init, stream, mr);
      case aggregation::MIN: return min(col, output_dtype, init, stream, mr);
      case aggregation::MAX: return max(col, output_dtype, init, stream, mr);
      case aggregation::ANY: return any(col, output_dtype, init, stream, mr);
      case aggregation::ALL: return all(col, output_dtype, init, stream, mr);
      case aggregation::HISTOGRAM: return histogram(col, stream, mr);
      case aggregation::MERGE_HISTOGRAM: return merge_histogram(col, stream, mr);
      case aggregation::SUM_OF_SQUARES: return sum_of_squares(col, output_dtype, stream, mr);
      case aggregation::MEAN: return mean(col, output_dtype, stream, mr);
      case aggregation::VARIANCE: {
        auto var_agg = static_cast<cudf::detail::var_aggregation const&>(agg);
        return variance(col, output_dtype, var_agg._ddof, stream, mr);
      }
      case aggregation::STD: {
        auto var_agg = static_cast<cudf::detail::std_aggregation const&>(agg);
        return standard_deviation(col, output_dtype, var_agg._ddof, stream, mr);
      }
      case aggregation::MEDIAN: {
        auto current_mr     = cudf::get_current_device_resource_ref();
        auto sorted_indices = cudf::detail::sorted_order(
          table_view{{col}}, {}, {null_order::AFTER}, stream, current_mr);
        auto valid_sorted_indices =
          cudf::detail::split(*sorted_indices, {col.size() - col.null_count()}, stream)[0];
        auto col_ptr = cudf::detail::quantile(
          col, {0.5}, interpolation::LINEAR, valid_sorted_indices, true, stream, current_mr);
        return cudf::detail::get_element(*col_ptr, 0, stream, mr);
      }
      case aggregation::QUANTILE: {
        auto quantile_agg = static_cast<cudf::detail::quantile_aggregation const&>(agg);
        CUDF_EXPECTS(quantile_agg._quantiles.size() == 1,
                     "Reduction quantile accepts only one quantile value");
        auto current_mr     = cudf::get_current_device_resource_ref();
        auto sorted_indices = cudf::detail::sorted_order(
          table_view{{col}}, {}, {null_order::AFTER}, stream, current_mr);
        auto valid_sorted_indices =
          cudf::detail::split(*sorted_indices, {col.size() - col.null_count()}, stream)[0];

        auto col_ptr = cudf::detail::quantile(col,
                                              quantile_agg._quantiles,
                                              quantile_agg._interpolation,
                                              valid_sorted_indices,
                                              true,
                                              stream,
                                              current_mr);
        return cudf::detail::get_element(*col_ptr, 0, stream, mr);
      }
      case aggregation::NUNIQUE: {
        auto nunique_agg = static_cast<cudf::detail::nunique_aggregation const&>(agg);
        return cudf::make_fixed_width_scalar(
          cudf::detail::distinct_count(
            col, nunique_agg._null_handling, nan_policy::NAN_IS_VALID, stream),
          stream,
          mr);
      }
      case aggregation::NTH_ELEMENT: {
        auto nth_agg = static_cast<cudf::detail::nth_element_aggregation const&>(agg);
        return nth_element(col, nth_agg._n, nth_agg._null_handling, stream, mr);
      }
      case aggregation::COLLECT_LIST: {
        auto col_agg = static_cast<cudf::detail::collect_list_aggregation const&>(agg);
        return collect_list(col, col_agg._null_handling, stream, mr);
      }
      case aggregation::COLLECT_SET: {
        auto col_agg = static_cast<cudf::detail::collect_set_aggregation const&>(agg);
        return collect_set(
          col, col_agg._null_handling, col_agg._nulls_equal, col_agg._nans_equal, stream, mr);
      }
      case aggregation::MERGE_LISTS: {
        return merge_lists(col, stream, mr);
      }
      case aggregation::MERGE_SETS: {
        auto col_agg = static_cast<cudf::detail::merge_sets_aggregation const&>(agg);
        return merge_sets(col, col_agg._nulls_equal, col_agg._nans_equal, stream, mr);
      }
      case aggregation::TDIGEST: {
        CUDF_EXPECTS(output_dtype.id() == type_id::STRUCT,
                     "Tdigest aggregations expect output type to be STRUCT");
        auto td_agg = static_cast<cudf::detail::tdigest_aggregation const&>(agg);
        return tdigest::detail::reduce_tdigest(col, td_agg.max_centroids, stream, mr);
      }
      case aggregation::MERGE_TDIGEST: {
        CUDF_EXPECTS(output_dtype.id() == type_id::STRUCT,
                     "Tdigest aggregations expect output type to be STRUCT");
        auto td_agg = static_cast<cudf::detail::merge_tdigest_aggregation const&>(agg);
        return tdigest::detail::reduce_merge_tdigest(col, td_agg.max_centroids, stream, mr);
      }
      default: CUDF_FAIL("Unsupported reduction operator");
    }
  }
};

std::unique_ptr<scalar> reduce(column_view const& col,
                               reduce_aggregation const& agg,
                               data_type output_dtype,
                               std::optional<std::reference_wrapper<scalar const>> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!init.has_value() || cudf::have_same_types(col, init.value().get()),
               "column and initial value must be the same type",
               cudf::data_type_error);
  if (init.has_value() && !(agg.kind == aggregation::SUM || agg.kind == aggregation::PRODUCT ||
                            agg.kind == aggregation::MIN || agg.kind == aggregation::MAX ||
                            agg.kind == aggregation::ANY || agg.kind == aggregation::ALL)) {
    CUDF_FAIL(
      "Initial value is only supported for SUM, PRODUCT, MIN, MAX, ANY, and ALL aggregation types");
  }

  // Returns default scalar if input column is empty or all null
  if (col.size() <= col.null_count()) {
    if (agg.kind == aggregation::TDIGEST || agg.kind == aggregation::MERGE_TDIGEST) {
      return tdigest::detail::make_empty_tdigest_scalar(stream, mr);
    }

    if (agg.kind == aggregation::HISTOGRAM) {
      return std::make_unique<list_scalar>(
        std::move(*reduction::detail::make_empty_histogram_like(col)), true, stream, mr);
    }
    if (agg.kind == aggregation::MERGE_HISTOGRAM) {
      return std::make_unique<list_scalar>(
        std::move(*reduction::detail::make_empty_histogram_like(col.child(0))), true, stream, mr);
    }

    if (agg.kind == aggregation::COLLECT_LIST || agg.kind == aggregation::COLLECT_SET) {
      auto scalar = make_list_scalar(empty_like(col)->view(), stream, mr);
      scalar->set_valid_async(false, stream);
      return scalar;
    }

    // `make_default_constructed_scalar` does not support nested type.
    if (cudf::is_nested(output_dtype)) { return make_empty_scalar_like(col, stream, mr); }

    auto result = make_default_constructed_scalar(output_dtype, stream, mr);
    if (agg.kind == aggregation::ANY || agg.kind == aggregation::ALL) {
      // empty input should return false for ANY and return true for ALL
      dynamic_cast<numeric_scalar<bool>*>(result.get())
        ->set_value(agg.kind == aggregation::ALL, stream);
    }
    return result;
  }

  return cudf::detail::aggregation_dispatcher(
    agg.kind, reduce_dispatch_functor{col, output_dtype, init, stream, mr}, agg);
}
}  // namespace detail
}  // namespace reduction

std::unique_ptr<scalar> reduce(column_view const& col,
                               reduce_aggregation const& agg,
                               data_type output_dtype,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return reduction::detail::reduce(col, agg, output_dtype, std::nullopt, stream, mr);
}

std::unique_ptr<scalar> reduce(column_view const& col,
                               reduce_aggregation const& agg,
                               data_type output_dtype,
                               std::optional<std::reference_wrapper<scalar const>> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return reduction::detail::reduce(col, agg, output_dtype, init, stream, mr);
}
}  // namespace cudf

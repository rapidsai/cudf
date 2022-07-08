/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {
struct reduce_dispatch_functor {
  column_view const col;
  data_type output_dtype;
  rmm::mr::device_memory_resource* mr;
  rmm::cuda_stream_view stream;

  reduce_dispatch_functor(column_view const& col,
                          data_type output_dtype,
                          rmm::cuda_stream_view stream,
                          rmm::mr::device_memory_resource* mr)
    : col(col), output_dtype(output_dtype), mr(mr), stream(stream)
  {
  }

  template <aggregation::Kind k>
  std::unique_ptr<scalar> operator()(std::unique_ptr<reduce_aggregation> const& agg)
  {
    switch (k) {
      case aggregation::SUM: return reduction::sum(col, output_dtype, stream, mr); break;
      case aggregation::PRODUCT: return reduction::product(col, output_dtype, stream, mr); break;
      case aggregation::MIN: return reduction::min(col, output_dtype, stream, mr); break;
      case aggregation::MAX: return reduction::max(col, output_dtype, stream, mr); break;
      case aggregation::ANY: return reduction::any(col, output_dtype, stream, mr); break;
      case aggregation::ALL: return reduction::all(col, output_dtype, stream, mr); break;
      case aggregation::SUM_OF_SQUARES:
        return reduction::sum_of_squares(col, output_dtype, stream, mr);
        break;
      case aggregation::MEAN: return reduction::mean(col, output_dtype, stream, mr); break;
      case aggregation::VARIANCE: {
        auto var_agg = dynamic_cast<var_aggregation const*>(agg.get());
        return reduction::variance(col, output_dtype, var_agg->_ddof, stream, mr);
      } break;
      case aggregation::STD: {
        auto var_agg = dynamic_cast<std_aggregation const*>(agg.get());
        return reduction::standard_deviation(col, output_dtype, var_agg->_ddof, stream, mr);
      } break;
      case aggregation::MEDIAN: {
        auto sorted_indices = sorted_order(table_view{{col}}, {}, {null_order::AFTER}, stream);
        auto valid_sorted_indices =
          split(*sorted_indices, {col.size() - col.null_count()}, stream)[0];
        auto col_ptr =
          quantile(col, {0.5}, interpolation::LINEAR, valid_sorted_indices, true, stream);
        return get_element(*col_ptr, 0, stream, mr);
      } break;
      case aggregation::QUANTILE: {
        auto quantile_agg = dynamic_cast<quantile_aggregation const*>(agg.get());
        CUDF_EXPECTS(quantile_agg->_quantiles.size() == 1,
                     "Reduction quantile accepts only one quantile value");
        auto sorted_indices = sorted_order(table_view{{col}}, {}, {null_order::AFTER}, stream);
        auto valid_sorted_indices =
          split(*sorted_indices, {col.size() - col.null_count()}, stream)[0];

        auto col_ptr = quantile(col,
                                quantile_agg->_quantiles,
                                quantile_agg->_interpolation,
                                valid_sorted_indices,
                                true,
                                stream);
        return get_element(*col_ptr, 0, stream, mr);
      } break;
      case aggregation::NUNIQUE: {
        auto nunique_agg = dynamic_cast<nunique_aggregation const*>(agg.get());
        return make_fixed_width_scalar(
          detail::distinct_count(
            col, nunique_agg->_null_handling, nan_policy::NAN_IS_VALID, stream),
          stream,
          mr);
      } break;
      case aggregation::NTH_ELEMENT: {
        auto nth_agg = dynamic_cast<nth_element_aggregation const*>(agg.get());
        return reduction::nth_element(col, nth_agg->_n, nth_agg->_null_handling, stream, mr);
      } break;
      case aggregation::COLLECT_LIST: {
        auto col_agg = dynamic_cast<collect_list_aggregation const*>(agg.get());
        return reduction::collect_list(col, col_agg->_null_handling, stream, mr);
      } break;
      case aggregation::COLLECT_SET: {
        auto col_agg = dynamic_cast<collect_set_aggregation const*>(agg.get());
        return reduction::collect_set(
          col, col_agg->_null_handling, col_agg->_nulls_equal, col_agg->_nans_equal, stream, mr);
      } break;
      case aggregation::MERGE_LISTS: {
        return reduction::merge_lists(col, stream, mr);
      } break;
      case aggregation::MERGE_SETS: {
        auto col_agg = dynamic_cast<merge_sets_aggregation const*>(agg.get());
        return reduction::merge_sets(col, col_agg->_nulls_equal, col_agg->_nans_equal, stream, mr);
      } break;
      case aggregation::TDIGEST: {
        CUDF_EXPECTS(output_dtype.id() == type_id::STRUCT,
                     "Tdigest aggregations expect output type to be STRUCT");
        auto td_agg = dynamic_cast<tdigest_aggregation const*>(agg.get());
        return detail::tdigest::reduce_tdigest(col, td_agg->max_centroids, stream, mr);
      } break;
      case aggregation::MERGE_TDIGEST: {
        CUDF_EXPECTS(output_dtype.id() == type_id::STRUCT,
                     "Tdigest aggregations expect output type to be STRUCT");
        auto td_agg = dynamic_cast<merge_tdigest_aggregation const*>(agg.get());
        return detail::tdigest::reduce_merge_tdigest(col, td_agg->max_centroids, stream, mr);
      } break;
      default: CUDF_FAIL("Unsupported reduction operator");
    }
  }
};

std::unique_ptr<scalar> reduce(
  column_view const& col,
  std::unique_ptr<reduce_aggregation> const& agg,
  data_type output_dtype,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  // Returns default scalar if input column is non-valid. In terms of nested columns, we need to
  // handcraft the default scalar with input column.
  if (col.size() <= col.null_count()) {
    if (agg->kind == aggregation::TDIGEST || agg->kind == aggregation::MERGE_TDIGEST) {
      return detail::tdigest::make_empty_tdigest_scalar();
    }
    if (col.type().id() == type_id::EMPTY || col.type() != output_dtype) {
      // Under some circumstance, the output type will become the List of input type,
      // such as: collect_list or collect_set. So, we have to handcraft the default scalar.
      if (output_dtype.id() == type_id::LIST) {
        auto scalar = make_list_scalar(empty_like(col)->view(), stream, mr);
        scalar->set_valid_async(false, stream);
        return scalar;
      }

      return make_default_constructed_scalar(output_dtype, stream, mr);
    }

    return make_empty_scalar_like(col, stream, mr);
  }

  return aggregation_dispatcher(
    agg->kind, reduce_dispatch_functor{col, output_dtype, stream, mr}, agg);
}
}  // namespace detail

std::unique_ptr<scalar> reduce(column_view const& col,
                               std::unique_ptr<reduce_aggregation> const& agg,
                               data_type output_dtype,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::reduce(col, agg, output_dtype, cudf::default_stream_value, mr);
}

}  // namespace cudf

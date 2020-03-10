/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/detail/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>

namespace cudf {
namespace experimental {
namespace detail {

template <bool has_nulls, typename ReturnType = bool>
struct unique_comparator {
  unique_comparator(table_device_view device_table,
                    size_type const *sorted_order)
      : comp(device_table, device_table, true), perm(sorted_order) {}
  __device__ ReturnType operator()(size_type index) const noexcept {
    return index == 0 || not comp(perm[index], perm[index - 1]);
  };

private:
  row_equality_comparator<has_nulls> comp;
  size_type const *perm;
};

std::unique_ptr<table> rank(table_view const &input, rank_method method,
                            order column_order, include_nulls _include_nulls,
                            null_order null_precedence,
                            bool percentage,
                            rmm::mr::device_memory_resource *mr,
                            cudaStream_t stream = 0) {
  auto const size = input.num_rows();

  std::vector<std::unique_ptr<column>> rank_columns;
  for (auto const &input_col : input) {
    std::unique_ptr<column> sorted_order =
        (method == rank_method::FIRST)
            ? detail::stable_sorted_order(table_view{{input_col}},
                                          {column_order}, {null_precedence}, mr,
                                          stream)
            : detail::sorted_order(table_view{{input_col}}, {column_order},
                                   {null_precedence}, mr, stream);
    column_view sorted_order_view = sorted_order->view();

    // na_option=keep assign NA to NA values
    if (_include_nulls == include_nulls::NO)
      rank_columns.push_back(make_numeric_column(
          data_type(FLOAT64), size, copy_bitmask(input_col, stream, mr),
          input_col.null_count(), stream, mr));
    else
      rank_columns.push_back(make_numeric_column(
          data_type(FLOAT64), size, mask_state::UNALLOCATED, stream, mr));

    auto rank_mutable_view = rank_columns.back()->mutable_view();
    auto rank_data = rank_mutable_view.data<double>();
    auto device_table =
        table_device_view::create(table_view{{input_col}}, stream);
    rmm::device_vector<size_type> dense_rank_sorted; // as key for min, max,
                                                     // average
    if (method != rank_method::FIRST) {
      dense_rank_sorted = rmm::device_vector<size_type>(size);
      if (input_col.has_nulls()) {
        auto conv = unique_comparator<true, size_type>(
            *device_table, sorted_order_view.data<size_type>());
        auto it = thrust::make_transform_iterator(
            thrust::make_counting_iterator<size_type>(0), conv);
        thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream), it,
                               it + size,
                               dense_rank_sorted.data().get());
      } else {
        auto conv = unique_comparator<false, size_type>(
            *device_table, sorted_order_view.data<size_type>());
        auto it = thrust::make_transform_iterator(
            thrust::make_counting_iterator<size_type>(0), conv);
        thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream), it,
                               it + size,
                               dense_rank_sorted.data().get());
      }
    }

    switch (method) {
    case rank_method::FIRST:
      thrust::scatter(
          rmm::exec_policy(stream)->on(stream),
          thrust::make_counting_iterator<double>(1),
          thrust::make_counting_iterator<double>(size + 1),
          sorted_order_view.begin<size_type>(), rank_data);
      break;
    case rank_method::DENSE:
      thrust::scatter(rmm::exec_policy(stream)->on(stream),
                      dense_rank_sorted.begin(), dense_rank_sorted.end(),
                      sorted_order_view.begin<size_type>(), rank_data);
      break;
    case rank_method::MIN: {
      rmm::device_vector<double> min_sorted(size, 0);
      thrust::reduce_by_key(
          rmm::exec_policy(stream)->on(stream), dense_rank_sorted.begin(),
          dense_rank_sorted.end(), thrust::make_counting_iterator<double>(1),
          thrust::make_discard_iterator(), min_sorted.begin(),
          thrust::equal_to<double>{}, thrust::minimum<double>{});
      auto sorted_min_rank = thrust::make_transform_iterator(
          dense_rank_sorted.begin(), [min_rank = min_sorted.begin()] 
          __device__( auto i) -> double { return min_rank[i - 1]; });
      thrust::scatter(rmm::exec_policy(stream)->on(stream), sorted_min_rank,
                      sorted_min_rank + size,
                      sorted_order_view.begin<size_type>(), rank_data);
    } break;
    case rank_method::MAX: {
      rmm::device_vector<double> max_sorted(size, 0);
      thrust::reduce_by_key(
          rmm::exec_policy(stream)->on(stream), dense_rank_sorted.begin(),
          dense_rank_sorted.end(), thrust::make_counting_iterator<double>(1),
          thrust::make_discard_iterator(), max_sorted.begin(),
          thrust::equal_to<double>{}, thrust::maximum<double>{});
      auto sorted_max_rank = thrust::make_transform_iterator(
          dense_rank_sorted.begin(), [max_rank = max_sorted.begin()] 
          __device__( auto i) -> double { return max_rank[i - 1]; });
      thrust::scatter(rmm::exec_policy(stream)->on(stream), sorted_max_rank,
                      sorted_max_rank + size,
                      sorted_order_view.begin<size_type>(), rank_data);
    } break;
    case rank_method::AVERAGE: {
      using MinCount = thrust::tuple<size_type, size_type>;
      rmm::device_vector<MinCount> min_count(size);
      thrust::reduce_by_key(
          rmm::exec_policy(stream)->on(stream), dense_rank_sorted.begin(),
          dense_rank_sorted.end(),
          thrust::make_zip_iterator(
              thrust::make_tuple(thrust::make_counting_iterator<size_type>(1),
                                 thrust::make_constant_iterator<size_type>(1))),
          thrust::make_discard_iterator(), min_count.begin(),
          thrust::equal_to<size_type>{}, [] __device__(auto i, auto j) {
            return MinCount{std::min(thrust::get<0>(i), thrust::get<0>(j)),
                            thrust::get<1>(i) + thrust::get<1>(j)};
          });
      auto avgit = thrust::make_transform_iterator(
          min_count.begin(),
          [] __device__(auto i) { // min+(count-1)/2
            return static_cast<double>(thrust::get<0>(i)) +
                   (static_cast<double>(thrust::get<1>(i)) - 1) / 2.0;
          });
      auto sorted_mean_rank = thrust::make_transform_iterator(
          dense_rank_sorted.begin(),
          [avgit] __device__(auto i) { return avgit[i - 1]; });
      thrust::scatter(rmm::exec_policy(stream)->on(stream), sorted_mean_rank,
                      sorted_mean_rank + size,
                      sorted_order_view.begin<size_type>(), rank_data);
    } break;
    default:
      CUDF_FAIL("Unexpected rank_method for rank()");
    }
    
    // pct inplace transform
    if (percentage) {
      size_type const count = (_include_nulls == include_nulls::NO)
                                          ? size - input_col.null_count()
                                          : size;
      auto drs = dense_rank_sorted.data().get();
      bool is_dense = (method == rank_method::DENSE);
      thrust::transform(rmm::exec_policy(stream)->on(stream), rank_data,
                        rank_data + size, rank_data,
                        [is_dense, drs, count] __device__(double r) -> double {
                          return is_dense? r/drs[count-1] : r/count;
                        });
    }
  }
  return std::make_unique<table>(std::move(rank_columns));
}
} // namespace detail

std::unique_ptr<table> rank(table_view input, rank_method method,
                            order column_order, include_nulls _include_nulls,
                            null_order null_precedence,
                            bool percentage,
                            rmm::mr::device_memory_resource *mr) {
  return detail::rank(input, method, column_order, _include_nulls,
                      null_precedence, percentage, mr);
}
} // namespace experimental
} // namespace cudf

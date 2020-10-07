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
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sequence.h>

namespace cudf {
namespace detail {
namespace {
// Functor to identify unique elements in a sorted order table/column
template <bool has_nulls, typename ReturnType, typename Iterator>
struct unique_comparator {
  unique_comparator(table_device_view device_table, Iterator const sorted_order)
    : comparator(device_table, device_table, true), permute(sorted_order)
  {
  }
  __device__ ReturnType operator()(size_type index) const noexcept
  {
    return index == 0 || not comparator(permute[index], permute[index - 1]);
  };

 private:
  row_equality_comparator<has_nulls> comparator;
  Iterator const permute;
};

// Assign rank from 1 to n unique values. Equal values get same rank value.
rmm::device_vector<size_type> sorted_dense_rank(column_view input_col,
                                                column_view sorted_order_view,
                                                cudaStream_t stream)
{
  auto device_table     = table_device_view::create(table_view{{input_col}}, stream);
  auto const input_size = input_col.size();
  rmm::device_vector<size_type> dense_rank_sorted(input_size);
  auto sorted_index_order = thrust::make_permutation_iterator(
    sorted_order_view.begin<size_type>(), thrust::make_counting_iterator<size_type>(0));
  if (input_col.has_nulls()) {
    auto conv = unique_comparator<true, size_type, decltype(sorted_index_order)>(
      *device_table, sorted_index_order);
    auto unique_it =
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), conv);

    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           unique_it,
                           unique_it + input_size,
                           dense_rank_sorted.data().get());
  } else {
    auto conv = unique_comparator<false, size_type, decltype(sorted_index_order)>(
      *device_table, sorted_index_order);
    auto unique_it =
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), conv);

    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           unique_it,
                           unique_it + input_size,
                           dense_rank_sorted.data().get());
  }
  return dense_rank_sorted;
}

/**
 * @brief Breaks the ties among equal value groups using binary operator and
 * transform this tied value to final rank.
 *
 * @param dense_rank dense rank of sorted input column (acts as key for value
 * groups).
 * @param tie_iter  iterator of rank to break ties among equal value groups.
 * @param sorted_order_view sorted order indices of input column
 * @param rank_iter output rank iterator
 * @param tie_breaker tie breaking operator. For example, maximum & minimum.
 * @param transformer transform after tie breaking (useful for average).
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename TieType,
          typename outputIterator,
          typename TieBreaker,
          typename Transformer,
          typename TieIterator>
void tie_break_ranks_transform(rmm::device_vector<size_type> const &dense_rank_sorted,
                               TieIterator tie_iter,
                               column_view const &sorted_order_view,
                               outputIterator rank_iter,
                               TieBreaker tie_breaker,
                               Transformer transformer,
                               cudaStream_t stream)
{
  auto const input_size = sorted_order_view.size();
  rmm::device_vector<TieType> tie_sorted(input_size, 0);
  // algorithm: reduce_by_key(dense_rank, 1, n, reduction_tie_breaker)
  // reduction_tie_breaker = min, max, min_count
  thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                        dense_rank_sorted.begin(),
                        dense_rank_sorted.end(),
                        tie_iter,
                        thrust::make_discard_iterator(),
                        tie_sorted.begin(),
                        thrust::equal_to<size_type>{},
                        tie_breaker);
  auto sorted_tied_rank = thrust::make_transform_iterator(
    dense_rank_sorted.begin(),
    [tied_rank = tie_sorted.begin(), transformer] __device__(auto dense_pos) {
      return transformer(tied_rank[dense_pos - 1]);
    });
  thrust::scatter(rmm::exec_policy(stream)->on(stream),
                  sorted_tied_rank,
                  sorted_tied_rank + input_size,
                  sorted_order_view.begin<size_type>(),
                  rank_iter);
}

template <typename outputType>
void rank_first(column_view sorted_order_view,
                mutable_column_view rank_mutable_view,
                cudaStream_t stream)
{
  // stable sort order ranking (no ties)
  thrust::scatter(rmm::exec_policy(stream)->on(stream),
                  thrust::make_counting_iterator<size_type>(1),
                  thrust::make_counting_iterator<size_type>(rank_mutable_view.size() + 1),
                  sorted_order_view.begin<size_type>(),
                  rank_mutable_view.begin<outputType>());
}

template <typename outputType>
void rank_dense(rmm::device_vector<size_type> const &dense_rank_sorted,
                column_view sorted_order_view,
                mutable_column_view rank_mutable_view,
                cudaStream_t stream)
{
  // All equal values have same rank and rank always increases by 1 between groups
  thrust::scatter(rmm::exec_policy(stream)->on(stream),
                  dense_rank_sorted.begin(),
                  dense_rank_sorted.end(),
                  sorted_order_view.begin<size_type>(),
                  rank_mutable_view.begin<outputType>());
}

template <typename outputType>
void rank_min(rmm::device_vector<size_type> const &group_keys,
              column_view sorted_order_view,
              mutable_column_view rank_mutable_view,
              cudaStream_t stream)
{
  // min of first in the group
  // All equal values have min of ranks among them.
  // algorithm: reduce_by_key(dense_rank, 1, n, min), scatter
  tie_break_ranks_transform<size_type>(group_keys,
                                       thrust::make_counting_iterator<size_type>(1),
                                       sorted_order_view,
                                       rank_mutable_view.begin<outputType>(),
                                       thrust::minimum<size_type>{},
                                       thrust::identity<outputType>{},
                                       stream);
}

template <typename outputType>
void rank_max(rmm::device_vector<size_type> const &group_keys,
              column_view sorted_order_view,
              mutable_column_view rank_mutable_view,
              cudaStream_t stream)
{
  // max of first in the group
  // All equal values have max of ranks among them.
  // algorithm: reduce_by_key(dense_rank, 1, n, max), scatter
  tie_break_ranks_transform<size_type>(group_keys,
                                       thrust::make_counting_iterator<size_type>(1),
                                       sorted_order_view,
                                       rank_mutable_view.begin<outputType>(),
                                       thrust::maximum<size_type>{},
                                       thrust::identity<outputType>{},
                                       stream);
}

void rank_average(rmm::device_vector<size_type> const &group_keys,
                  column_view sorted_order_view,
                  mutable_column_view rank_mutable_view,
                  cudaStream_t stream)
{
  // k, k+1, .. k+n-1
  // average = (n*k+ n*(n-1)/2)/n
  // average = k + (n-1)/2 = min + (count-1)/2
  // Calculate Min of ranks and Count of equal values
  // algorithm: reduce_by_key(dense_rank, 1, n, min_count)
  //            transform(min+(count-1)/2), scatter
  using MinCount = thrust::tuple<size_type, size_type>;
  tie_break_ranks_transform<MinCount>(
    group_keys,
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator<size_type>(1),
                                                 thrust::make_constant_iterator<size_type>(1))),
    sorted_order_view,
    rank_mutable_view.begin<double>(),
    [] __device__(auto rank_count1, auto rank_count2) {
      return MinCount{std::min(thrust::get<0>(rank_count1), thrust::get<0>(rank_count2)),
                      thrust::get<1>(rank_count1) + thrust::get<1>(rank_count2)};
    },
    [] __device__(MinCount minrank_count) {  // min+(count-1)/2
      return static_cast<double>(thrust::get<0>(minrank_count)) +
             (static_cast<double>(thrust::get<1>(minrank_count)) - 1) / 2.0;
    },
    stream);
}

}  // anonymous namespace

std::unique_ptr<column> rank(column_view const &input,
                             rank_method method,
                             order column_order,
                             null_policy null_handling,
                             null_order null_precedence,
                             bool percentage,
                             rmm::mr::device_memory_resource *mr,
                             cudaStream_t stream = 0)
{
  data_type const output_type = (percentage or method == rank_method::AVERAGE)
                                  ? data_type(type_id::FLOAT64)
                                  : data_type(type_to_id<size_type>());
  std::unique_ptr<column> rank_column = [&null_handling, &output_type, &input, &mr, &stream] {
    // na_option=keep assign NA to NA values
    if (null_handling == null_policy::EXCLUDE)
      return make_numeric_column(
        output_type, input.size(), copy_bitmask(input, stream, mr), input.null_count(), stream, mr);
    else
      return make_numeric_column(output_type, input.size(), mask_state::UNALLOCATED, stream, mr);
  }();
  auto rank_mutable_view = rank_column->mutable_view();

  std::unique_ptr<column> sorted_order =
    (method == rank_method::FIRST)
      ? detail::stable_sorted_order(
          table_view{{input}}, {column_order}, {null_precedence}, mr, stream)
      : detail::sorted_order(table_view{{input}}, {column_order}, {null_precedence}, mr, stream);
  column_view sorted_order_view = sorted_order->view();

  // dense: All equal values have same rank and rank always increases by 1 between groups
  // acts as key for min, max, average to denote equal value groups
  rmm::device_vector<size_type> const dense_rank_sorted =
    [&method, &input, &sorted_order_view, &stream] {
      if (method != rank_method::FIRST)
        return sorted_dense_rank(input, sorted_order_view, stream);
      else
        return rmm::device_vector<size_type>();
    }();

  if (output_type.id() == type_id::FLOAT64) {
    switch (method) {
      case rank_method::FIRST:
        rank_first<double>(sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::DENSE:
        rank_dense<double>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::MIN:
        rank_min<double>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::MAX:
        rank_max<double>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::AVERAGE:
        rank_average(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      default: CUDF_FAIL("Unexpected rank_method for rank()");
    }
  } else {
    switch (method) {
      case rank_method::FIRST:
        rank_first<size_type>(sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::DENSE:
        rank_dense<size_type>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::MIN:
        rank_min<size_type>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::MAX:
        rank_max<size_type>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::AVERAGE:
        rank_average(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      default: CUDF_FAIL("Unexpected rank_method for rank()");
    }
  }

  // pct inplace transform
  if (percentage) {
    auto rank_iter = rank_mutable_view.begin<double>();
    size_type const count =
      (null_handling == null_policy::EXCLUDE) ? input.size() - input.null_count() : input.size();
    auto drs            = dense_rank_sorted.data().get();
    bool const is_dense = (method == rank_method::DENSE);
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      rank_iter,
                      rank_iter + input.size(),
                      rank_iter,
                      [is_dense, drs, count] __device__(double r) -> double {
                        return is_dense ? r / drs[count - 1] : r / count;
                      });
  }
  return rank_column;
}
}  // namespace detail

std::unique_ptr<column> rank(column_view const &input,
                             rank_method method,
                             order column_order,
                             null_policy null_handling,
                             null_order null_precedence,
                             bool percentage,
                             rmm::mr::device_memory_resource *mr)
{
  return detail::rank(input, method, column_order, null_handling, null_precedence, percentage, mr);
}
}  // namespace cudf

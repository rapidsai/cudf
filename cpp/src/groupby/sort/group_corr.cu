/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <memory>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <type_traits>
#include "cudf/types.hpp"
#include "groupby/sort/group_reductions.hpp"
#include "thrust/functional.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/zip_iterator.h"

#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <typename T>
constexpr bool is_double_convertible()
{
  return std::is_convertible_v<T, double> || std::is_constructible_v<double, T>;
}

struct is_double_convertible_impl {
  template <typename T>
  bool operator()()
  {
    return is_double_convertible<T>();
  }
};

/**
 * @brief Type casts each element of the column to `CastType`
 *
 */
template <typename CastType>
struct type_casted_accessor {
  template <typename Element>
  CUDA_DEVICE_CALLABLE CastType operator()(cudf::size_type i, column_device_view const& col) const
  {
    if constexpr (column_device_view::has_element_accessor<Element>() and
                  std::is_convertible_v<Element, CastType>)
      return static_cast<CastType>(col.element<Element>(i));
    return {};
  }
};

template <typename ResultType>
struct corr_transform {  // : thrust::unary_function<size_type, ResultType>
  column_device_view const d_values_0, d_values_1;
  ResultType const *d_means_0, *d_means_1;
  ResultType const *d_stddev_0, *d_stddev_1;
  size_type const* d_group_sizes;
  size_type const* d_group_labels;
  size_type ddof{1};  // TODO update based on bias.

  __device__ ResultType operator()(size_type i)
  {
    if (d_values_0.is_null(i) or d_values_1.is_null(i)) return 0.0;

    // This has to be device dispatch because x and y type may differ
    auto x = type_dispatcher(d_values_0.type(), type_casted_accessor<ResultType>{}, i, d_values_0);
    auto y = type_dispatcher(d_values_1.type(), type_casted_accessor<ResultType>{}, i, d_values_1);

    size_type group_idx  = d_group_labels[i];
    size_type group_size = d_group_sizes[group_idx];

    // prevent divide by zero error
    if (group_size == 0 or group_size - ddof <= 0) return 0.0;

    ResultType xmean   = d_means_0[group_idx];
    ResultType ymean   = d_means_1[group_idx];
    ResultType xstddev = d_stddev_0[group_idx];
    ResultType ystddev = d_stddev_1[group_idx];
    return (x - xmean) * (y - ymean) / (group_size - ddof) / xstddev / ystddev;
  }
};
}  // namespace

// TODO Eventually this function should accept values_0, values_1, not a struct.
std::unique_ptr<column> group_corr(column_view const& values_0,
                                   column_view const& values_1,
                                   cudf::device_span<size_type const> group_offsets,
                                   cudf::device_span<size_type const> group_labels,
                                   size_type num_groups,
                                   column_view const& mean_0,
                                   column_view const& mean_1,
                                   column_view const& stddev_0,
                                   column_view const& stddev_1,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  using result_type = id_to_type<type_id::FLOAT64>;
  static_assert(
    std::is_same_v<cudf::detail::target_type_t<result_type, aggregation::Kind::CORRELATION>,
                   result_type>);

  // check if each child type can be converted to float64.
  bool const is_convertible = type_dispatcher(values_0.type(), is_double_convertible_impl{}) or
                              type_dispatcher(values_1.type(), is_double_convertible_impl{});

  CUDF_EXPECTS(is_convertible,
               "Input to `group_corr` must be columns of type convertible to float64.");

  // TODO calculate COUNT_VALID  (need to do for 2 seperately. for MEAN, and
  // bitmask_and->COUNT_VALID for CORR.)
  // TODO calculate CORR. (requires MEAN1, MEAN2, COUNT_VALID_ANDed, STDDEV1, STDDEV2)
  // TODO shuffle.

  auto mean0_ptr   = mean_0.begin<result_type>();
  auto mean1_ptr   = mean_1.begin<result_type>();
  auto stddev0_ptr = stddev_0.begin<result_type>();
  auto stddev1_ptr = stddev_1.begin<result_type>();

  // TODO replace with ANDed bitmask. (values, stddev)
  auto count1 = values_0.nullable()
                  ? detail::group_count_valid(values_0, group_labels, num_groups, stream, mr)
                  : detail::group_count_all(group_offsets, num_groups, stream, mr);

  auto d_values_0 = column_device_view::create(values_0, stream);
  auto d_values_1 = column_device_view::create(values_1, stream);
  corr_transform<result_type> corr_transform_op{*d_values_0,
                                                *d_values_1,
                                                mean0_ptr,
                                                mean1_ptr,
                                                stddev0_ptr,
                                                stddev1_ptr,
                                                count1->view().data<size_type>(),
                                                group_labels.begin()};

  // result
  auto const any_nulls = values_0.has_nulls() or values_1.has_nulls();
  auto mask_type       = any_nulls ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED;

  auto result =
    make_numeric_column(data_type(type_to_id<result_type>()), num_groups, mask_type, stream, mr);
  auto d_result = result->mutable_view().begin<result_type>();

  auto corr_iter =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0), corr_transform_op);

  thrust::reduce_by_key(rmm::exec_policy(stream),
                        group_labels.begin(),
                        group_labels.end(),
                        corr_iter,
                        thrust::make_discard_iterator(),
                        d_result);
  return result;

  // auto result_M2s = make_numeric_column(
  //   data_type(type_to_id<result_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  // auto validities = rmm::device_uvector<int8_t>(num_groups, stream);

  // // Perform merging for all the aggregations. Their output (and their validity data) are written
  // // out concurrently through an output zip iterator.
  // using iterator_tuple  = thrust::tuple<size_type*, result_type*, result_type*, int8_t*>;
  // using output_iterator = thrust::zip_iterator<iterator_tuple>;
  // auto const out_iter =
  //   output_iterator{thrust::make_tuple(result_counts->mutable_view().template data<size_type>(),
  //                                      result_means->mutable_view().template data<result_type>(),
  //                                      result_M2s->mutable_view().template data<result_type>(),
  //                                      validities.begin())};

  // auto const count_valid = values.child(0);
  // auto const mean_values = values.child(1);
  // auto const M2_values   = values.child(2);
  // auto const iter        = thrust::make_counting_iterator<size_type>(0);

  // auto const fn = merge_fn<result_type>{group_offsets.begin(),
  //                                       count_valid.template begin<size_type>(),
  //                                       mean_values.template begin<result_type>(),
  //                                       M2_values.template begin<result_type>()};
  // thrust::transform(rmm::exec_policy(stream), iter, iter + num_groups, out_iter, fn);

  // // Generate bitmask for the output.
  // // Only mean and M2 values can be nullable. Count column must be non-nullable.
  // auto [null_mask, null_count] = cudf::detail::valid_if(
  //   validities.begin(), validities.end(), thrust::identity<int8_t>{}, stream, mr);
  // if (null_count > 0) {
  //   result_means->set_null_mask(null_mask, null_count);           // copy null_mask
  //   result_M2s->set_null_mask(std::move(null_mask), null_count);  // take over null_mask
  // }

  // Output is a structs column containing the merged values of `COUNT_VALID`, `MEAN`, and `M2`.

  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf

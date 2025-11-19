/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <typename ResultType, typename Iterator>
struct m2_transform {
  column_device_view const d_values;
  Iterator const values_iter;
  ResultType const* d_means;
  size_type const* d_group_labels;

  __device__ ResultType operator()(size_type const idx) const noexcept
  {
    if (d_values.is_null(idx)) { return 0.0; }

    auto const x         = static_cast<ResultType>(values_iter[idx]);
    auto const group_idx = d_group_labels[idx];
    auto const mean      = d_means[group_idx];
    auto const diff      = x - mean;
    return diff * diff;
  }
};

template <typename ResultType, typename Iterator>
void compute_m2_fn(column_device_view const& values,
                   Iterator values_iter,
                   cudf::device_span<size_type const> group_labels,
                   ResultType const* d_means,
                   ResultType* d_result,
                   rmm::cuda_stream_view stream)
{
  auto m2_fn = m2_transform<ResultType, decltype(values_iter)>{
    values, values_iter, d_means, group_labels.data()};
  auto const itr = thrust::counting_iterator<size_type>(0);
  // Using a temporary buffer for intermediate transform results instead of
  // using the transform-iterator directly in thrust::reduce_by_key
  // improves compile-time significantly.
  auto m2_vals = rmm::device_uvector<ResultType>(values.size(), stream);
  thrust::transform(rmm::exec_policy(stream), itr, itr + values.size(), m2_vals.begin(), m2_fn);

  thrust::reduce_by_key(rmm::exec_policy(stream),
                        group_labels.begin(),
                        group_labels.end(),
                        m2_vals.begin(),
                        thrust::make_discard_iterator(),
                        d_result);
}

struct m2_functor {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& values,
                                     column_view const& group_means,
                                     cudf::device_span<size_type const> group_labels,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(std::is_arithmetic_v<T>)
  {
    using result_type = cudf::detail::target_type_t<T, aggregation::Kind::M2>;
    auto result       = make_numeric_column(data_type(type_to_id<result_type>()),
                                      group_means.size(),
                                      mask_state::UNALLOCATED,
                                      stream,
                                      mr);

    auto const values_dv_ptr = column_device_view::create(values, stream);
    auto const d_values      = *values_dv_ptr;
    auto const d_means       = group_means.data<result_type>();
    auto const d_result      = result->mutable_view().data<result_type>();

    if (!cudf::is_dictionary(values.type())) {
      auto const values_iter = d_values.begin<T>();
      compute_m2_fn(d_values, values_iter, group_labels, d_means, d_result, stream);
    } else {
      auto const values_iter =
        cudf::dictionary::detail::make_dictionary_iterator<T>(*values_dv_ptr);
      compute_m2_fn(d_values, values_iter, group_labels, d_means, d_result, stream);
    }

    // M2 column values should have the same bitmask as means's.
    if (group_means.nullable()) {
      result->set_null_mask(cudf::detail::copy_bitmask(group_means, stream, mr),
                            group_means.null_count());
    }

    return result;
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(!std::is_arithmetic_v<T>)
  {
    CUDF_FAIL("Only numeric types are supported in M2 groupby aggregation");
  }
};

}  // namespace

std::unique_ptr<column> group_m2(column_view const& values,
                                 column_view const& group_means,
                                 cudf::device_span<size_type const> group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto values_type = cudf::is_dictionary(values.type())
                       ? dictionary_column_view(values).keys().type()
                       : values.type();

  return type_dispatcher(values_type, m2_functor{}, values, group_means, group_labels, stream, mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf

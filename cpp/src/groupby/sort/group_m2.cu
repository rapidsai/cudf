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
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <typename FloatType, typename Iterator>
struct m2_transform {
  column_device_view const d_values;
  Iterator const d_values_iter;
  FloatType const* d_means;
  size_type const* d_group_labels;

  __device__ FloatType operator()(size_type const idx) const noexcept
  {
    if (d_values.is_null(idx)) { return 0.0; }

    auto const x         = static_cast<FloatType>(d_values_iter[idx]);
    auto const group_idx = d_group_labels[idx];
    auto const mean      = d_means[group_idx];
    auto const diff      = x - mean;
    return diff * diff;
  }
};

template <typename FloatType, typename Iterator>
void compute_m2_fn(column_device_view const& d_values,
                   Iterator d_values_iter,
                   cudf::device_span<size_type const> group_labels,
                   FloatType const* d_means,
                   FloatType* d_m2s,
                   rmm::cuda_stream_view stream)
{
  auto const var_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0},
    m2_transform<FloatType, Iterator>{d_values, d_values_iter, d_means, group_labels.data()});

  thrust::reduce_by_key(rmm::exec_policy(stream),
                        group_labels.begin(),
                        group_labels.end(),
                        var_iter,
                        thrust::make_discard_iterator(),
                        d_m2s);
}

struct m2_functor {
  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, std::unique_ptr<column>> operator()(
    column_view const& values,
    column_view const& group_means,
    cudf::device_span<size_type const> group_labels,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    // Output double type for m2 values.
    using float_type = id_to_type<type_id::FLOAT64>;
    CUDF_EXPECTS(group_means.type().id() == type_to_id<float_type>(),
                 "Input `group_means` column must have double type.");

    auto m2s = make_numeric_column(
      data_type(type_to_id<float_type>()), group_means.size(), mask_state::UNALLOCATED, stream, mr);

    auto const values_dv_ptr = column_device_view::create(values, stream);
    auto const d_values      = *values_dv_ptr;
    auto const d_means       = group_means.data<float_type>();
    auto const d_m2s         = m2s->mutable_view().data<float_type>();

    if (!cudf::is_dictionary(values.type())) {
      auto const d_values_iter = d_values.template begin<T>();
      compute_m2_fn(d_values, d_values_iter, group_labels, d_means, d_m2s, stream);
    } else {
      auto const d_values_iter =
        cudf::dictionary::detail::make_dictionary_iterator<T>(*values_dv_ptr);
      compute_m2_fn(d_values, d_values_iter, group_labels, d_means, d_m2s, stream);
    }

    // M2 column values should have the same bitmask as means's.
    if (group_means.nullable()) {
      m2s->set_null_mask(cudf::detail::copy_bitmask(group_means, stream, mr),
                         group_means.null_count());
    }

    return m2s;
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic<T>::value, std::unique_ptr<column>> operator()(Args&&...)
  {
    CUDF_FAIL("Only numeric types are supported in M2 groupby aggregation");
  }
};

}  // namespace

std::unique_ptr<column> group_m2(column_view const& values,
                                 column_view const& group_counts,
                                 column_view const& group_means,
                                 cudf::device_span<size_type const> group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  auto values_type = cudf::is_dictionary(values.type())
                       ? dictionary_column_view(values).keys().type()
                       : values.type();

  // Firstly compute m2 values.
  auto m2s =
    type_dispatcher(values_type, m2_functor{}, values, group_means, group_labels, stream, mr);

  // Then build the output structs column having double members (count, mean, m2).
  std::vector<std::unique_ptr<column>> output_members;
  output_members.emplace_back(cudf::detail::cast(group_counts, group_means.type(), stream, mr));
  output_members.emplace_back(std::make_unique<column>(group_means, stream, mr));
  output_members.emplace_back(std::move(m2s));

  return make_structs_column(group_counts.size(),
                             std::move(output_members),
                             0,
                             rmm::device_buffer{0, stream, mr},
                             stream,
                             mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf

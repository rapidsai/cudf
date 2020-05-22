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

#include <memory>
#include <vector>

#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <quantiles/quantiles_util.hpp>

namespace cudf {
namespace detail {
template <bool exact, typename SortMapIterator>
struct quantile_functor {
  SortMapIterator ordered_indices;
  size_type size;
  std::vector<double> const& q;
  interpolation interp;
  bool retain_types;
  rmm::mr::device_memory_resource* mr;
  cudaStream_t stream;

  template <typename T>
  std::enable_if_t<not std::is_arithmetic<T>::value, std::unique_ptr<column>> operator()(
    column_view const& input)
  {
    CUDF_FAIL("quantile does not support non-numeric types");
  }

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, std::unique_ptr<column>> operator()(
    column_view const& input)
  {
    using Result = std::conditional_t<exact, double, T>;

    auto type   = data_type{type_to_id<Result>()};
    auto output = make_fixed_width_column(type, q.size(), mask_state::UNALLOCATED, stream, mr);

    if (output->size() == 0) { return output; }

    if (input.size() == 0) {
      auto mask = create_null_mask(output->size(), mask_state::ALL_NULL, stream, mr);
      output->set_null_mask(std::move(mask), output->size());
      return output;
    }

    auto d_input  = column_device_view::create(input);
    auto d_output = mutable_column_device_view::create(output->mutable_view());

    rmm::device_vector<double> q_device{q};

    auto sorted_data = thrust::make_permutation_iterator(input.data<T>(), ordered_indices);

    thrust::transform(q_device.begin(),
                      q_device.end(),
                      d_output->template begin<Result>(),
                      [sorted_data, interp = interp, size = size] __device__(double q) {
                        return select_quantile_data<Result>(sorted_data, size, q, interp);
                      });

    if (input.nullable()) {
      auto sorted_validity = thrust::make_transform_iterator(
        ordered_indices,
        [input = *d_input] __device__(size_type idx) { return input.is_valid_nocheck(idx); });

      rmm::device_buffer mask;
      size_type null_count;

      std::tie(mask, null_count) = valid_if(
        q_device.begin(),
        q_device.end(),
        [sorted_validity, interp = interp, size = size] __device__(double q) {
          return select_quantile_validity(sorted_validity, size, q, interp);
        },
        stream,
        mr);

      output->set_null_mask(std::move(mask), null_count);
    }

    return output;
  }
};

template <bool exact, typename SortMapIterator>
std::unique_ptr<column> quantile(column_view const& input,
                                 SortMapIterator ordered_indices,
                                 size_type size,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 bool retain_types,
                                 rmm::mr::device_memory_resource* mr,
                                 cudaStream_t stream)
{
  auto functor = quantile_functor<exact, SortMapIterator>{
    ordered_indices, size, q, interp, retain_types, mr, stream};

  return type_dispatcher(input.type(), functor, input);
}

}  // namespace detail

std::unique_ptr<column> quantile(column_view const& input,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 column_view const& ordered_indices,
                                 bool exact,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (ordered_indices.is_empty()) {
    if (exact) {
      return detail::quantile<true>(
        input, thrust::make_counting_iterator<size_type>(0), input.size(), q, interp, exact, mr, 0);
    } else {
      return detail::quantile<false>(
        input, thrust::make_counting_iterator<size_type>(0), input.size(), q, interp, exact, mr, 0);
    }

  } else {
    CUDF_EXPECTS(ordered_indices.type() == data_type{type_to_id<size_type>()},
                 "`ordered_indicies` type must be `INT32`.");

    if (exact) {
      return detail::quantile<true>(
        input, ordered_indices.data<size_type>(), ordered_indices.size(), q, interp, exact, mr, 0);
    } else {
      return detail::quantile<false>(
        input, ordered_indices.data<size_type>(), ordered_indices.size(), q, interp, exact, mr, 0);
    }
  }
}

}  // namespace cudf
